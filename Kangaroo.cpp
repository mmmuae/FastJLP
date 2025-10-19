/*
 * This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
 * Copyright (c) 2020 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Kangaroo.h"
#include <fstream>
#include "SECPK1/IntGroup.h"
#include "Timer.h"
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <memory>
#ifdef WITHGPU
#include "GPU/BackendFactory.h"
#endif
#ifndef WIN64
#include <pthread.h>
#endif

using namespace std;

#define safe_delete_array(x) if(x) {delete[] x;x=NULL;}

#if defined(GPU_BACKEND_METAL)
#include "GPU/metal/MetalPacking.h"
#include "GPU/metal/MetalDistinguishedPoint.h"
#endif  // GPU_BACKEND_METAL

#ifdef WITHGPU
extern BackendKind gRequestedBackend;
#endif

// ----------------------------------------------------------------------------

Kangaroo::Kangaroo(Secp256K1 *secp,int32_t initDPSize,bool useGpu,string &workFile,string &iWorkFile,uint32_t savePeriod,bool saveKangaroo,bool saveKangarooByServer,
                   double maxStep,int wtimeout,int port,int ntimeout,string serverIp,string outputFile,bool splitWorkfile) {

  this->secp = secp;
  this->initDPSize = initDPSize;
  this->useGpu = useGpu;
  this->offsetCount = 0;
  this->offsetTime = 0.0;
  this->workFile = workFile;
  this->saveWorkPeriod = savePeriod;
  this->inputFile = iWorkFile;
  this->nbLoadedWalk = 0;
  this->clientMode = serverIp.length() > 0;
  this->saveKangarooByServer = this->clientMode && saveKangarooByServer;
  this->saveKangaroo = saveKangaroo || this->saveKangarooByServer;
  this->fRead = NULL;
  this->maxStep = maxStep;
  this->wtimeout = wtimeout;
  this->port = port;
  this->ntimeout = ntimeout;
  this->serverIp = serverIp;
  this->outputFile = outputFile;
  this->hostInfo = NULL;
  this->endOfSearch = false;
  this->saveRequest = false;
  this->connectedClient = 0;
  this->totalRW = 0;
  this->collisionInSameHerd = 0;
  this->keyIdx = 0;
  this->splitWorkfile = splitWorkfile;
  this->pid = Timer::getPID();

  CPU_GRP_SIZE = 1024;

  // Init mutex
#ifdef WIN64
  ghMutex = CreateMutex(NULL,FALSE,NULL);
  saveMutex = CreateMutex(NULL,FALSE,NULL);
#else
  pthread_mutex_init(&ghMutex, NULL);
  pthread_mutex_init(&saveMutex, NULL);
  signal(SIGPIPE, SIG_IGN);
#endif

}

// ----------------------------------------------------------------------------

bool Kangaroo::ParseConfigFile(std::string &fileName) {

  // In client mode, config come from the server
  if(clientMode)
    return true;

  // Check file
  FILE *fp = fopen(fileName.c_str(),"rb");
  if(fp == NULL) {
    ::printf("Error: Cannot open %s %s\n",fileName.c_str(),strerror(errno));
    return false;
  }
  fclose(fp);

  // Get lines
  vector<string> lines;
  string line;
  ifstream inFile(fileName);
  while(getline(inFile,line)) {

    // Remove ending \r\n
    int l = (int)line.length() - 1;
    while(l >= 0 && isspace(line.at(l))) {
      line.pop_back();
      l--;
    }

    if(line.length() > 0) {
      lines.push_back(line);
    }

  }

  if(lines.size()<3) {
    ::printf("Error: %s not enough arguments\n",fileName.c_str());
    return false;
  }

  rangeStart.SetBase16(lines[0].c_str());
  rangeEnd.SetBase16(lines[1].c_str());
  for(int i=2;i<(int)lines.size();i++) {
    
    Point p;
    bool isCompressed;
    if( !secp->ParsePublicKeyHex(lines[i],p,isCompressed) ) {
      ::printf("%s, error line %d: %s\n",fileName.c_str(),i,lines[i].c_str());
      return false;
    }
    keysToSearch.push_back(p);

  }

  ::printf("Start:%s\n",rangeStart.GetBase16().c_str());
  ::printf("Stop :%s\n",rangeEnd.GetBase16().c_str());
  ::printf("Keys :%d\n",(int)keysToSearch.size());

  return true;

}

// ----------------------------------------------------------------------------

bool Kangaroo::IsDP(uint64_t x) {

  return (x & dMask) == 0;

}

void Kangaroo::SetDP(int size) {

  // Mask for distinguised point
  dpSize = size;
  if(dpSize == 0) {
    dMask = 0;
  } else {
    if(dpSize > 64) dpSize = 64;
    dMask = (1ULL << (64 - dpSize)) - 1;
    dMask = ~dMask;
  }

#ifdef WIN64
  ::printf("DP size: %d [0x%016I64X]\n",dpSize,dMask);
#else
  ::printf("DP size: %d [0x%" PRIx64 "]\n",dpSize,dMask);
#endif

}

// ----------------------------------------------------------------------------

bool Kangaroo::Output(Int *pk,char sInfo,int sType) {


  FILE* f = stdout;
  bool needToClose = false;

  if(outputFile.length() > 0) {
    f = fopen(outputFile.c_str(),"a");
    if(f == NULL) {
      printf("Cannot open %s for writing\n",outputFile.c_str());
      f = stdout;
    }
    else {
      needToClose = true;
    }
  }

  if(!needToClose)
    ::printf("\n");

  Point PR = secp->ComputePublicKey(pk);

  ::fprintf(f,"Key#%2d [%d%c]Pub:  0x%s \n",keyIdx,sType,sInfo,secp->GetPublicKeyHex(true,keysToSearch[keyIdx]).c_str());
  if(PR.equals(keysToSearch[keyIdx])) {
    ::fprintf(f,"       Priv: 0x%s \n",pk->GetBase16().c_str());
  } else {
    ::fprintf(f,"       Failed !\n");
    if(needToClose)
      fclose(f);
    return false;
  }


  if(needToClose)
    fclose(f);

  return true;

}

// ----------------------------------------------------------------------------

bool  Kangaroo::CheckKey(Int d1,Int d2,uint8_t type) {

  // Resolve equivalence collision

  if(type & 0x1)
    d1.ModNegK1order();
  if(type & 0x2)
    d2.ModNegK1order();

  Int pk(&d1);
  pk.ModAddK1order(&d2);

  Point P = secp->ComputePublicKey(&pk);

  if(P.equals(keyToSearch)) {
    // Key solved    
#ifdef USE_SYMMETRY
    pk.ModAddK1order(&rangeWidthDiv2);
#endif
    pk.ModAddK1order(&rangeStart);    
    return Output(&pk,'N',type);
  }

  if(P.equals(keyToSearchNeg)) {
    // Key solved
    pk.ModNegK1order();
#ifdef USE_SYMMETRY
    pk.ModAddK1order(&rangeWidthDiv2);
#endif
    pk.ModAddK1order(&rangeStart);
    return Output(&pk,'S',type);
  }

  return false;

}

bool Kangaroo::CollisionCheck(Int* d1,uint32_t type1,Int* d2,uint32_t type2) {


  if(type1 == type2) {

    // Collision inside the same herd
    return false;

  } else {

    Int Td;
    Int Wd;

    if(type1 == TAME) {
      Td.Set(d1);
      Wd.Set(d2);
    }  else {
      Td.Set(d2);
      Wd.Set(d1);
    }

    endOfSearch = CheckKey(Td,Wd,0) || CheckKey(Td,Wd,1) || CheckKey(Td,Wd,2) || CheckKey(Td,Wd,3);

    if(!endOfSearch) {

      // Should not happen, reset the kangaroo
      ::printf("\n Unexpected wrong collision, reset kangaroo !\n");
      if((int64_t)(Td.bits64[3])<0) {
        Td.ModNegK1order();
        ::printf("Found: Td-%s\n",Td.GetBase16().c_str());
      } else {
        ::printf("Found: Td %s\n",Td.GetBase16().c_str());
      }
      if((int64_t)(Wd.bits64[3])<0) {
        Wd.ModNegK1order();
        ::printf("Found: Wd-%s\n",Wd.GetBase16().c_str());
      } else {
        ::printf("Found: Wd %s\n",Wd.GetBase16().c_str());
      }
      return false;

    }

  }

  return true;

}

// ----------------------------------------------------------------------------

bool Kangaroo::AddToTable(Int *pos,Int *dist,uint32_t kType) {

  int addStatus = hashTable.Add(pos,dist,kType);
  if(addStatus== ADD_COLLISION)
    return CollisionCheck(&hashTable.kDist,hashTable.kType,dist,kType);

  return addStatus == ADD_OK;

}

bool Kangaroo::AddToTable(uint64_t h,int128_t *x,int128_t *d) {

  int addStatus = hashTable.Add(h,x,d);
  if(addStatus== ADD_COLLISION) {

    Int dist;
    uint32_t kType;
    HashTable::CalcDistAndType(*d,&dist,&kType);
    return CollisionCheck(&hashTable.kDist,hashTable.kType,&dist,kType);

  }

  return addStatus == ADD_OK;

}

// ----------------------------------------------------------------------------

void Kangaroo::SolveKeyCPU(TH_PARAM *ph) {

  vector<ITEM> dps;
  double lastSent = 0;

  // Global init
  int thId = ph->threadId;

  // Create Kangaroos
  ph->nbKangaroo = CPU_GRP_SIZE;

#ifdef USE_SYMMETRY
  ph->symClass = new uint64_t[CPU_GRP_SIZE];
  for(int i = 0; i<CPU_GRP_SIZE; i++) ph->symClass[i] = 0;
#endif

  IntGroup *grp = new IntGroup(CPU_GRP_SIZE);
  Int *dx = new Int[CPU_GRP_SIZE];

  if(ph->px==NULL) {

    // Create Kangaroos, if not already loaded
    ph->px = new Int[CPU_GRP_SIZE];
    ph->py = new Int[CPU_GRP_SIZE];
    ph->distance = new Int[CPU_GRP_SIZE];
    CreateHerd(CPU_GRP_SIZE,ph->px,ph->py,ph->distance,TAME);

  }

  if(keyIdx==0)
    ::printf("SolveKeyCPU Thread %d: %d kangaroos\n",ph->threadId,CPU_GRP_SIZE);

  ph->hasStarted = true;

  // Using Affine coord
  Int dy;
  Int rx;
  Int ry;
  Int _s;
  Int _p;

  while(!endOfSearch) {

    // Random walk

    for(int g = 0; g < CPU_GRP_SIZE; g++) {

#ifdef USE_SYMMETRY
      uint64_t jmp = ph->px[g].bits64[0] % (NB_JUMP/2) + (NB_JUMP / 2) * ph->symClass[g];
#else
      uint64_t jmp = ph->px[g].bits64[0] % NB_JUMP;
#endif

      Int *p1x = &jumpPointx[jmp];
      Int *p2x = &ph->px[g];
      dx[g].ModSub(p2x,p1x);

    }

    grp->Set(dx);
    grp->ModInv();

    for(int g = 0; g < CPU_GRP_SIZE; g++) {

#ifdef USE_SYMMETRY
      uint64_t jmp = ph->px[g].bits64[0] % (NB_JUMP / 2) + (NB_JUMP / 2) * ph->symClass[g];
#else
      uint64_t jmp = ph->px[g].bits64[0] % NB_JUMP;
#endif

      Int *p1x = &jumpPointx[jmp];
      Int *p1y = &jumpPointy[jmp];
      Int *p2x = &ph->px[g];
      Int *p2y = &ph->py[g];

      dy.ModSub(p2y,p1y);
      _s.ModMulK1(&dy,&dx[g]);
      _p.ModSquareK1(&_s);

      rx.ModSub(&_p,p1x);
      rx.ModSub(p2x);

      ry.ModSub(p2x,&rx);
      ry.ModMulK1(&_s);
      ry.ModSub(p2y);

      ph->distance[g].ModAddK1order(&jumpDistance[jmp]);

#ifdef USE_SYMMETRY
      // Equivalence symmetry class switch
      if( ry.ModPositiveK1() ) {
        ph->distance[g].ModNegK1order();
        ph->symClass[g] = !ph->symClass[g];
      }
#endif

      ph->px[g].Set(&rx);
      ph->py[g].Set(&ry);

    }

    if( clientMode ) {

      // Send DP to server
      for(int g = 0; g < CPU_GRP_SIZE; g++) {
        if(IsDP(ph->px[g].bits64[3])) {
          ITEM it;
          it.x.Set(&ph->px[g]);
          it.d.Set(&ph->distance[g]);
          it.kIdx = g;
          dps.push_back(it);
        }
      }

      double now = Timer::get_tick();
      if( now-lastSent > SEND_PERIOD ) {
        LOCK(ghMutex);
        SendToServer(dps,ph->threadId,0xFFFF);
        UNLOCK(ghMutex);
        lastSent = now;
      }

      if(!endOfSearch) counters[thId] += CPU_GRP_SIZE;

    } else {

      // Add to table and collision check
      for(int g = 0; g < CPU_GRP_SIZE && !endOfSearch; g++) {

        if(IsDP(ph->px[g].bits64[3])) {
          LOCK(ghMutex);
          if(!endOfSearch) {

            if(!AddToTable(&ph->px[g],&ph->distance[g],g % 2)) {
              // Collision inside the same herd
              // We need to reset the kangaroo
              CreateHerd(1,&ph->px[g],&ph->py[g],&ph->distance[g],g % 2,false);
              collisionInSameHerd++;
            }

          }
          UNLOCK(ghMutex);
        }

        if(!endOfSearch) counters[thId] ++;

      }

    }

    // Save request
    if(saveRequest && !endOfSearch) {
      ph->isWaiting = true;
      LOCK(saveMutex);
      ph->isWaiting = false;
      UNLOCK(saveMutex);
    }

  }

  // Free
  delete grp;
  delete[] dx;
  safe_delete_array(ph->px);
  safe_delete_array(ph->py);
  safe_delete_array(ph->distance);
#ifdef USE_SYMMETRY
  safe_delete_array(ph->symClass);
#endif

  ph->isRunning = false;

}

// ----------------------------------------------------------------------------

void Kangaroo::SolveKeyGPU(TH_PARAM *ph) {

#ifdef WITHGPU
  int thId = ph->threadId;
  double lastSent = 0.0;
  bool handledBackend = false;
#if defined(GPU_BACKEND_METAL)
  if (!handledBackend && gRequestedBackend == BackendKind::METAL) {
    handledBackend = true;

  vector<ITEM> dps;
  vector<ITEM> gpuFound;
  constexpr uint32_t kMaxFound = 65536 * 2;
#ifdef USE_SYMMETRY
  const Int& wildOffset = rangeWidthDiv4;
#else
  const Int& wildOffset = rangeWidthDiv2;
#endif

  auto backendDeleter = [](IGpuBackend* backend) {
    if (backend) {
      backend->shutdown();
      delete backend;
    }
  };

  std::unique_ptr<IGpuBackend, decltype(backendDeleter)> backend(
      CreateBackend(BackendKind::METAL), backendDeleter);

  if (!backend) {
    ::printf("Metal backend: CreateBackend failed\n");
    ph->hasStarted = true;
    return;
  }

  if (!backend->init()) {
    ::printf("Metal backend: init failed\n");
    ph->hasStarted = true;
    return;
  }

  double t0 = Timer::get_tick();

  if (ph->px == NULL) {
    if (keyIdx == 0)
      ::printf("SolveKeyGPU Thread Metal: creating kangaroos...\n");
    uint64_t nbThread = ph->nbKangaroo / GPU_GRP_SIZE;
    ph->px = new Int[ph->nbKangaroo];
    ph->py = new Int[ph->nbKangaroo];
    ph->distance = new Int[ph->nbKangaroo];

    for (uint64_t i = 0; i < nbThread; i++) {
      CreateHerd(GPU_GRP_SIZE,
                &(ph->px[i * GPU_GRP_SIZE]),
                &(ph->py[i * GPU_GRP_SIZE]),
                &(ph->distance[i * GPU_GRP_SIZE]),
                TAME);
    }
  }

  std::vector<MetalKangaroo> herd(ph->nbKangaroo);
  for (uint64_t i = 0; i < ph->nbKangaroo; ++i) {
    PackKangaroo(ph->px[i], ph->py[i], ph->distance[i], i, wildOffset, herd[i]);
  }

  std::vector<uint64_t> jumpDistBuf(NB_JUMP * 2ULL);
  std::vector<uint64_t> jumpPxBuf(NB_JUMP * 4ULL);
  std::vector<uint64_t> jumpPyBuf(NB_JUMP * 4ULL);
  for (int i = 0; i < NB_JUMP; ++i) {
    jumpDistBuf[i * 2 + 0] = jumpDistance[i].bits64[0];
    jumpDistBuf[i * 2 + 1] = jumpDistance[i].bits64[1];
    for (int limb = 0; limb < 4; ++limb) {
      jumpPxBuf[i * 4 + limb] = jumpPointx[i].bits64[limb];
      jumpPyBuf[i * 4 + limb] = jumpPointy[i].bits64[limb];
    }
  }

  alignas(16) uint64_t prime[4] = {0, 0, 0, 0};
  Int* primeInt = Int::GetFieldCharacteristic();
  for (int limb = 0; limb < 4; ++limb) {
    prime[limb] = primeInt->bits64[limb];
  }

  Buffers buffers{};
  buffers.kangaroos = herd.data();
  buffers.jumpDist = jumpDistBuf.data();
  buffers.jumpPx = jumpPxBuf.data();
  buffers.jumpPy = jumpPyBuf.data();
  buffers.dpItems = nullptr;
  buffers.prime = prime;
  buffers.dpCount = nullptr;
  buffers.totalKangaroos = static_cast<uint32_t>(ph->nbKangaroo);

  GpuConfig config{};
  config.threadsPerGroup = static_cast<uint32_t>(ph->gridSizeY);
  config.groups = static_cast<uint32_t>(ph->gridSizeX);
  config.iterationsPerDispatch = NB_RUN;
  config.jumpCount = NB_JUMP;
  config.dpMask = dMask;
  config.maxFound = kMaxFound;

  if (!backend->allocate(buffers, config)) {
    ::printf("Metal backend: allocate failed\n");
    ph->hasStarted = true;
    return;
  }

  if (!backend->uploadJumps(jumpDistBuf.data(), jumpPxBuf.data(), jumpPyBuf.data(), NB_JUMP)) {
    ::printf("Metal backend: uploadJumps failed\n");
    ph->hasStarted = true;
    return;
  }

  if (!backend->uploadKangaroos(herd.data(), herd.size() * sizeof(MetalKangaroo))) {
    ::printf("Metal backend: uploadKangaroos failed\n");
    ph->hasStarted = true;
    return;
  }

  backend->resetDPCount();

  if(workFile.length()==0 || !saveKangaroo) {
    safe_delete_array(ph->px);
    safe_delete_array(ph->py);
    safe_delete_array(ph->distance);
  }

  double t1 = Timer::get_tick();
  if (keyIdx == 0) {
    ::printf("SolveKeyGPU Thread Metal: 2^%.2f kangaroos [%.1fs]\n",
             log2((double)ph->nbKangaroo),(t1 - t0));
  }

  ph->hasStarted = true;

  std::vector<uint32_t> dpRing((static_cast<size_t>(kMaxFound) + 1ULL) * ITEM_SIZE32, 0);
  std::vector<MetalKangaroo> herdSnapshot(ph->nbKangaroo);

  while(!endOfSearch) {

    if (!backend->runOnce()) {
      ::printf("Metal backend: runOnce failed\n");
      break;
    }

    counters[thId] += ph->nbKangaroo * NB_RUN;

    uint32_t dpCount = 0;
    if (!backend->readDP(dpRing.data(), dpRing.size() * sizeof(uint32_t), dpCount)) {
      ::printf("Metal backend: readDP failed\n");
      break;
    }

    gpuFound.clear();
    for (uint32_t i = 0; i < dpCount; ++i) {
      const uint32_t* itemWords = dpRing.data() + i * ITEM_SIZE32 + 1;
      MetalDpItem decoded;
      MetalDecodeDistinguishedPoint(itemWords, decoded);
      ITEM it;
      it.x.SetInt32(0);
      it.d.SetInt32(0);
      for (int limb = 0; limb < 4; ++limb) {
        it.x.bits64[limb] = decoded.x.bits64[limb];
      }
      for (int limb = 0; limb < 2; ++limb) {
        it.d.bits64[limb] = decoded.dist.bits64[limb];
      }
      it.kIdx = decoded.index;
      if ((it.kIdx % 2) == WILD) {
        it.d.ModSubK1order(const_cast<Int*>(&wildOffset));
      }
      gpuFound.push_back(it);
    }

    if( clientMode ) {

      for(int i=0;i<(int)gpuFound.size();i++)
        dps.push_back(gpuFound[i]);

      double now = Timer::get_tick();
      if(now - lastSent > SEND_PERIOD) {
        LOCK(ghMutex);
        SendToServer(dps,ph->threadId,ph->gpuId);
        UNLOCK(ghMutex);
        lastSent = now;
      }

    } else {

      if(gpuFound.size() > 0) {

        LOCK(ghMutex);

        for(size_t g = 0; !endOfSearch && g < gpuFound.size(); g++) {

          uint32_t kType = (uint32_t)(gpuFound[g].kIdx % 2);

          if(!AddToTable(&gpuFound[g].x,&gpuFound[g].d,kType)) {
            Int pxNew;
            Int pyNew;
            Int dNew;
            CreateHerd(1,&pxNew,&pyNew,&dNew,kType,false);
            if (backend->downloadKangaroos(herdSnapshot.data(), herdSnapshot.size() * sizeof(MetalKangaroo))) {
              PackKangaroo(pxNew, pyNew, dNew, gpuFound[g].kIdx, wildOffset,
                           herdSnapshot[gpuFound[g].kIdx]);
              if (!backend->uploadKangaroos(herdSnapshot.data(), herdSnapshot.size() * sizeof(MetalKangaroo))) {
                ::printf("Metal backend: uploadKangaroos failed during reset\n");
              }
            } else {
              ::printf("Metal backend: downloadKangaroos failed during reset\n");
            }
            collisionInSameHerd++;
          }

        }
        UNLOCK(ghMutex);
      }

    }

    if(saveRequest && !endOfSearch) {
      if(saveKangaroo) {
        if(ph->px == NULL) {
          ph->px = new Int[ph->nbKangaroo];
          ph->py = new Int[ph->nbKangaroo];
          ph->distance = new Int[ph->nbKangaroo];
        }
        if (backend->downloadKangaroos(herdSnapshot.data(), herdSnapshot.size() * sizeof(MetalKangaroo))) {
          for (uint64_t i = 0; i < ph->nbKangaroo; ++i) {
            UnpackKangaroo(herdSnapshot[i], i, wildOffset,
                           ph->px[i], ph->py[i], ph->distance[i]);
          }
        } else {
          ::printf("Metal backend: downloadKangaroos failed during save\n");
        }
      }
      ph->isWaiting = true;
      LOCK(saveMutex);
      ph->isWaiting = false;
      UNLOCK(saveMutex);
    }

  }

  safe_delete_array(ph->px);
  safe_delete_array(ph->py);
  safe_delete_array(ph->distance);

  }

#endif
#if defined(GPU_BACKEND_CUDA)
  if (!handledBackend && gRequestedBackend == BackendKind::CUDA) {
    handledBackend = true;

  vector<ITEM> dps;
  vector<ITEM> gpuFound;
  GPUEngine *gpu;

  gpu = new GPUEngine(ph->gridSizeX,ph->gridSizeY,ph->gpuId,65536 * 2);

  if(keyIdx == 0)
    ::printf("GPU: %s (%.1f MB used)\n",gpu->deviceName.c_str(),gpu->GetMemory() / 1048576.0);

  double t0 = Timer::get_tick();


  if( ph->px==NULL ) {
    if(keyIdx == 0)
      ::printf("SolveKeyGPU Thread GPU#%d: creating kangaroos...\n",ph->gpuId);
    // Create Kangaroos, if not already loaded
    uint64_t nbThread = gpu->GetNbThread();
    ph->px = new Int[ph->nbKangaroo];
    ph->py = new Int[ph->nbKangaroo];
    ph->distance = new Int[ph->nbKangaroo];

    for(uint64_t i = 0; i<nbThread; i++) {
      CreateHerd(GPU_GRP_SIZE,&(ph->px[i*GPU_GRP_SIZE]),
                              &(ph->py[i*GPU_GRP_SIZE]),
                              &(ph->distance[i*GPU_GRP_SIZE]),
                              TAME);
    }
  }

#ifdef USE_SYMMETRY
  gpu->SetWildOffset(&rangeWidthDiv4);
#else
  gpu->SetWildOffset(&rangeWidthDiv2);
#endif
  gpu->SetParams(dMask,jumpDistance,jumpPointx,jumpPointy);
  gpu->SetKangaroos(ph->px,ph->py,ph->distance);

  if(workFile.length()==0 || !saveKangaroo) {
    // No need to get back kangaroo, free memory
    safe_delete_array(ph->px);
    safe_delete_array(ph->py);
    safe_delete_array(ph->distance);
  }

  gpu->callKernel();

  double t1 = Timer::get_tick();

  if(keyIdx == 0)
    ::printf("SolveKeyGPU Thread GPU#%d: 2^%.2f kangaroos [%.1fs]\n",ph->gpuId,log2((double)ph->nbKangaroo),(t1-t0));

  ph->hasStarted = true;

  while(!endOfSearch) {

    gpu->Launch(gpuFound);
    counters[thId] += ph->nbKangaroo * NB_RUN;

    if( clientMode ) {

      for(int i=0;i<(int)gpuFound.size();i++)
        dps.push_back(gpuFound[i]);

      double now = Timer::get_tick();
      if(now - lastSent > SEND_PERIOD) {
        LOCK(ghMutex);
        SendToServer(dps,ph->threadId,ph->gpuId);
        UNLOCK(ghMutex);
        lastSent = now;
      }

    } else {

      if(gpuFound.size() > 0) {

        LOCK(ghMutex);

        for(int g = 0; !endOfSearch && g < gpuFound.size(); g++) {

          uint32_t kType = (uint32_t)(gpuFound[g].kIdx % 2);

          if(!AddToTable(&gpuFound[g].x,&gpuFound[g].d,kType)) {
            // Collision inside the same herd
            // We need to reset the kangaroo
            Int px;
            Int py;
            Int d;
            CreateHerd(1,&px,&py,&d,kType,false);
            gpu->SetKangaroo(gpuFound[g].kIdx,&px,&py,&d);
            collisionInSameHerd++;
          }

        }
        UNLOCK(ghMutex);
      }

    }

    // Save request
    if(saveRequest && !endOfSearch) {
      // Get kangaroos
      if(saveKangaroo)
        gpu->GetKangaroos(ph->px,ph->py,ph->distance);
      ph->isWaiting = true;
      LOCK(saveMutex);
      ph->isWaiting = false;
      UNLOCK(saveMutex);
    }

  }


  safe_delete_array(ph->px);
  safe_delete_array(ph->py);
  safe_delete_array(ph->distance);
  delete gpu;

  }

#endif  // GPU_BACKEND_CUDA

  if (!handledBackend) {
    ::printf("Requested GPU backend '%s' is not available in this build\n",
             BackendName(gRequestedBackend));
    ph->hasStarted = true;
  }

#else

  ph->hasStarted = true;

#endif

  ph->isRunning = false;

}

// ----------------------------------------------------------------------------

#ifdef WIN64
DWORD WINAPI _SolveKeyCPU(LPVOID lpParam) {
#else
void *_SolveKeyCPU(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->SolveKeyCPU(p);
  return 0;
}

#ifdef WIN64
DWORD WINAPI _SolveKeyGPU(LPVOID lpParam) {
#else
void *_SolveKeyGPU(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->SolveKeyGPU(p);
  return 0;
}

// ----------------------------------------------------------------------------

void Kangaroo::CreateHerd(int nbKangaroo,Int *px,Int *py,Int *d,int firstType,bool lock) {

  vector<Int> pk;
  vector<Point> S;
  vector<Point> Sp;
  pk.reserve(nbKangaroo);
  S.reserve(nbKangaroo);
  Sp.reserve(nbKangaroo);
  Point Z;
  Z.Clear();

  // Choose random starting distance
  if(lock) LOCK(ghMutex);

  for(int j = 0; j < nbKangaroo; j++) {

#ifdef USE_SYMMETRY

    // Tame in [0..N/2]
    d[j].Rand(rangePower - 1);
    if((j+ firstType) % 2 == WILD) {
      // Wild in [-N/4..N/4]
      d[j].ModSubK1order(&rangeWidthDiv4);
    }

#else

    // Tame in [0..N]
    d[j].Rand(rangePower);
    if((j + firstType) % 2 == WILD) {
      // Wild in [-N/2..N/2]
      d[j].ModSubK1order(&rangeWidthDiv2);
    }

#endif

    pk.push_back(d[j]);

  }

  if(lock) UNLOCK(ghMutex);

  // Compute starting pos
  S = secp->ComputePublicKeys(pk);

  for(int j = 0; j < nbKangaroo; j++) {
    if((j + firstType) % 2 == TAME) {
      Sp.push_back(Z);
    } else {
      Sp.push_back(keyToSearch);
    }
  }

  S = secp->AddDirect(Sp,S);

  for(int j = 0; j < nbKangaroo; j++) {

    px[j].Set(&S[j].x);
    py[j].Set(&S[j].y);

#ifdef USE_SYMMETRY
    // Equivalence symmetry class switch
    if( py[j].ModPositiveK1() )
      d[j].ModNegK1order();
#endif

  }

}

// ----------------------------------------------------------------------------

void Kangaroo::CreateJumpTable() {

#ifdef USE_SYMMETRY
  int jumpBit = rangePower / 2;
#else
  int jumpBit = rangePower / 2 + 1;
#endif

  if(jumpBit > 128) jumpBit = 128;
  int maxRetry = 100;
  bool ok = false;
  double distAvg;
  double maxAvg = pow(2.0,(double)jumpBit - 0.95);
  double minAvg = pow(2.0,(double)jumpBit - 1.05);
  //::printf("Jump Avg distance min: 2^%.2f\n",log2(minAvg));
  //::printf("Jump Avg distance max: 2^%.2f\n",log2(maxAvg));
  
  // Kangaroo jumps
  // Constant seed for compatibilty of workfiles
  rseed(0x600DCAFE);

#ifdef USE_SYMMETRY
  Int old;
  old.Set(Int::GetFieldCharacteristic());
  Int u;
  Int v;
  u.SetInt32(1);
  u.ShiftL(jumpBit/2);
  u.AddOne();
  while(!u.IsProbablePrime()) {
    u.AddOne();
    u.AddOne();
  }
  v.Set(&u);
  v.AddOne();
  v.AddOne();
  while(!v.IsProbablePrime()) {
    v.AddOne();
    v.AddOne();
  }
  Int::SetupField(&old);

  ::printf("U= %s\n",u.GetBase16().c_str());
  ::printf("V= %s\n",v.GetBase16().c_str());
#endif

  // Positive only
  // When using symmetry, the sign is switched by the symmetry class switch
  while(!ok && maxRetry>0 ) {
    Int totalDist;
    totalDist.SetInt32(0);
#ifdef USE_SYMMETRY
    for(int i = 0; i < NB_JUMP/2; ++i) {
      jumpDistance[i].Rand(jumpBit/2);
      jumpDistance[i].Mult(&u);
      if(jumpDistance[i].IsZero())
        jumpDistance[i].SetInt32(1);
      totalDist.Add(&jumpDistance[i]);
    }
    for(int i = NB_JUMP / 2; i < NB_JUMP; ++i) {
      jumpDistance[i].Rand(jumpBit/2);
      jumpDistance[i].Mult(&v);
      if(jumpDistance[i].IsZero())
        jumpDistance[i].SetInt32(1);
      totalDist.Add(&jumpDistance[i]);
    }
#else
    for(int i = 0; i < NB_JUMP; ++i) {
      jumpDistance[i].Rand(jumpBit);
      if(jumpDistance[i].IsZero())
        jumpDistance[i].SetInt32(1);
      totalDist.Add(&jumpDistance[i]);
  }
#endif
    distAvg = totalDist.ToDouble() / (double)(NB_JUMP);
    ok = distAvg>minAvg && distAvg<maxAvg;
    maxRetry--;
  }

  for(int i = 0; i < NB_JUMP; ++i) {
    Point J = secp->ComputePublicKey(&jumpDistance[i]);
    jumpPointx[i].Set(&J.x);
    jumpPointy[i].Set(&J.y);
  }

  ::printf("Jump Avg distance: 2^%.2f\n",log2(distAvg));

  unsigned long seed = Timer::getSeed32();
  rseed(seed);

}

// ----------------------------------------------------------------------------

void Kangaroo::ComputeExpected(double dp,double *op,double *ram,double *overHead) {

  // Compute expected number of operation and memory

#ifdef USE_SYMMETRY
  double gainS = 1.0 / sqrt(2.0);
#else
  double gainS = 1.0;
#endif

  // Kangaroo number
  double k = (double)totalRW;

  // Range size
  double N = pow(2.0,(double)rangePower);

  // theta
  double theta = pow(2.0,dp);

  // Z0
  double Z0 = (2.0 * (2.0 - sqrt(2.0)) * gainS) * sqrt(M_PI);

  // Average for DP = 0
  double avgDP0 = Z0 * sqrt(N);

  // DP Overhead
  *op = Z0 * pow(N * (k * theta + sqrt(N)),1.0 / 3.0);

  *ram = (double)sizeof(HASH_ENTRY) * (double)HASH_SIZE + // Table
         (double)sizeof(ENTRY *) * (double)(HASH_SIZE * 4) + // Allocation overhead
         (double)(sizeof(ENTRY) + sizeof(ENTRY *)) * (*op / theta); // Entries

  *ram /= (1024.0*1024.0);

  if(overHead)
    *overHead = *op/avgDP0;

}

// ----------------------------------------------------------------------------

void Kangaroo::InitRange() {

  rangeWidth.Set(&rangeEnd);
  rangeWidth.Sub(&rangeStart);
  rangePower = rangeWidth.GetBitLength();
  ::printf("Range width: 2^%d\n",rangePower);
  rangeWidthDiv2.Set(&rangeWidth);
  rangeWidthDiv2.ShiftR(1);
  rangeWidthDiv4.Set(&rangeWidthDiv2);
  rangeWidthDiv4.ShiftR(1);
  rangeWidthDiv8.Set(&rangeWidthDiv4);
  rangeWidthDiv8.ShiftR(1);

}

void Kangaroo::InitSearchKey() {

  Int SP;
  SP.Set(&rangeStart);
#ifdef USE_SYMMETRY
  SP.ModAddK1order(&rangeWidthDiv2);
#endif
  if(!SP.IsZero()) {
    Point RS = secp->ComputePublicKey(&SP);
    RS.y.ModNeg();
    Point sum = secp->AddDirect(keysToSearch[keyIdx],RS);
    keyToSearch.Set(sum);
  } else {
    keyToSearch.Set(keysToSearch[keyIdx]);
  }
  keyToSearchNeg.Set(keyToSearch);
  keyToSearchNeg.y.ModNeg();

}

// ----------------------------------------------------------------------------

void Kangaroo::Run(int nbThread,std::vector<int> gpuId,std::vector<int> gridSize) {

  double t0 = Timer::get_tick();

  nbCPUThread = nbThread;
  nbGPUThread = (useGpu ? (int)gpuId.size() : 0);
#ifndef WITHGPU
  (void)gpuId;
  (void)gridSize;
#endif
  totalRW = 0;

#ifndef WITHGPU

  if(nbGPUThread>0) {
    ::printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
    nbGPUThread = 0;
  }

#endif

  uint64_t totalThread = (uint64_t)nbCPUThread + (uint64_t)nbGPUThread;
  if(totalThread == 0) {
    ::printf("No CPU or GPU thread, exiting.\n");
    ::exit(0);
  }

  TH_PARAM *params = (TH_PARAM *)malloc(totalThread * sizeof(TH_PARAM));
  THREAD_HANDLE *thHandles = (THREAD_HANDLE *)malloc(totalThread * sizeof(THREAD_HANDLE));

  memset(params, 0,totalThread * sizeof(TH_PARAM));
  memset(counters, 0, sizeof(counters));
  ::printf("Number of CPU thread: %d\n", nbCPUThread);

#ifdef WITHGPU

  // Compute grid size
  for(int i = 0; i < nbGPUThread; i++) {
    int x = gridSize[2ULL * i];
    int y = gridSize[2ULL * i + 1ULL];
#if defined(GPU_BACKEND_METAL) && !defined(GPU_BACKEND_CUDA)
    if(x <= 0) x = 64;
    if(y <= 0) y = 32;
    params[nbCPUThread + i].gridSizeX = x;
    params[nbCPUThread + i].gridSizeY = y;
    params[nbCPUThread + i].gpuId = 0;
#else
    if(!GPUEngine::GetGridSize(gpuId[i],&x,&y)) {
      return;
    } else {
      params[nbCPUThread + i].gridSizeX = x;
      params[nbCPUThread + i].gridSizeY = y;
    }
#endif
    params[nbCPUThread + i].nbKangaroo = (uint64_t)GPU_GRP_SIZE * x * y;
    totalRW += params[nbCPUThread + i].nbKangaroo;
  }

#endif

  totalRW += nbCPUThread * (uint64_t)CPU_GRP_SIZE;

  // Set starting parameters
  if( clientMode ) {
    // Retrieve config from server
    if( !GetConfigFromServer() )
      ::exit(0);
    // Client save only kangaroos, force -ws
    if(workFile.length()>0)
      saveKangaroo = true;
  }

  InitRange();
  CreateJumpTable();

  ::printf("Number of kangaroos: 2^%.2f\n",log2((double)totalRW));

  if( !clientMode ) {

    // Compute suggested distinguished bits number for less than 5% overhead (see README)
    double dpOverHead;
    int suggestedDP = (int)((double)rangePower / 2.0 - log2((double)totalRW));
    if(suggestedDP<0) suggestedDP=0;
    ComputeExpected((double)suggestedDP,&expectedNbOp,&expectedMem,&dpOverHead);
    while(dpOverHead>1.05 && suggestedDP>0) {
      suggestedDP--;
      ComputeExpected((double)suggestedDP,&expectedNbOp,&expectedMem,&dpOverHead);
    }

    if(initDPSize < 0)
      initDPSize = suggestedDP;

    ComputeExpected((double)initDPSize,&expectedNbOp,&expectedMem);
    if(nbLoadedWalk == 0) ::printf("Suggested DP: %d\n",suggestedDP);
    ::printf("Expected operations: 2^%.2f\n",log2(expectedNbOp));
    ::printf("Expected RAM: %.1fMB\n",expectedMem);

  } else {

    keyIdx = 0;
    InitSearchKey();

  }

  SetDP(initDPSize);

  // Fetch kangaroos (if any)
  FectchKangaroos(params);

//#define STATS
#ifdef STATS

    CPU_GRP_SIZE = 1024;
    for(; CPU_GRP_SIZE <= 1024; CPU_GRP_SIZE *= 4) {

      uint64_t totalCount = 0;
      uint64_t totalDead = 0;

#endif

    for(keyIdx = 0; keyIdx < keysToSearch.size(); keyIdx++) {

      InitSearchKey();

      endOfSearch = false;
      collisionInSameHerd = 0;

      // Reset conters
      memset(counters,0,sizeof(counters));

      // Lanch CPU threads
      for(int i = 0; i < nbCPUThread; i++) {
        params[i].threadId = i;
        params[i].isRunning = true;
        thHandles[i] = LaunchThread(_SolveKeyCPU,params + i);
      }

#ifdef WITHGPU

      // Launch GPU threads
      for(int i = 0; i < nbGPUThread; i++) {
        int id = nbCPUThread + i;
        params[id].threadId = 0x80L + i;
        params[id].isRunning = true;
        params[id].gpuId = gpuId[i];
        thHandles[id] = LaunchThread(_SolveKeyGPU,params + id);
      }

#endif


      // Wait for end
      Process(params,"MK/s");
      JoinThreads(thHandles,nbCPUThread + nbGPUThread);
      FreeHandles(thHandles,nbCPUThread + nbGPUThread);
      hashTable.Reset();

#ifdef STATS

      uint64_t count = getCPUCount() + getGPUCount();
      totalCount += count;
      totalDead += collisionInSameHerd;
      double SN = pow(2.0,rangePower / 2.0);
      double avg = (double)totalCount / (double)(keyIdx + 1);
      ::printf("\n[%3d] 2^%.3f Dead:%d Avg:2^%.3f DeadAvg:%.1f (%.3f %.3f sqrt(N))\n",
                              keyIdx, log2((double)count), (int)collisionInSameHerd, 
                              log2(avg), (double)totalDead / (double)(keyIdx + 1),
                              avg/SN,expectedNbOp/SN);
    }
    string fName = "DP" + ::to_string(dpSize) + ".txt";
    FILE *f = fopen(fName.c_str(),"a");
    fprintf(f,"%d %f\n",CPU_GRP_SIZE*nbCPUThread,(double)totalCount);
    fclose(f);

#endif

  }

  double t1 = Timer::get_tick();

  ::printf("\nDone: Total time %s \n" , GetTimeStr(t1-t0+offsetTime).c_str());

}


