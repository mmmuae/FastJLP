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
#include <iomanip>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <utility>
#ifndef WIN64
#include <pthread.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

using namespace std;

struct HashEntrySnapshot {
  int256_t x;
  int256_t d;
  uint32_t kType;
};

struct HashTableSnapshot {
  std::vector<uint32_t> bucketSizes;
  std::vector<uint32_t> bucketMax;
  std::vector<uint64_t> bucketOffsets;
  std::vector<HashEntrySnapshot> entries;
};

struct AsyncSavePayload {
  HashTableSnapshot tableSnapshot;
  std::vector<Int> kangarooX;
  std::vector<Int> kangarooY;
  std::vector<Int> kangarooD;
  std::vector<int256_t> kangaroosForServer;
  std::string rangeStartHex;
  std::string rangeEndHex;
  std::string keyXHex;
  std::string keyYHex;
  std::string fileName;
  std::string textFileName;
  bool hasBinaryTarget = false;
  bool hasTextTarget = false;
  bool needServerSend = false;
  bool saveKangaroo = false;
  bool saveKangarooText = false;
  bool splitWorkfile = false;
  uint64_t totalWalk = 0;
  uint64_t textKangarooCount = 0;
  uint32_t dpBits = 0;
  uint64_t totalCount = 0;
  double totalTime = 0.0;
  double startTick = 0.0;
  int headType = HEADW;
};

// ----------------------------------------------------------------------------

int Kangaroo::FSeek(FILE* stream,uint64_t pos) {

#ifdef WIN64
  return _fseeki64(stream,pos,SEEK_SET);
#else
  return fseeko(stream,pos,SEEK_SET);
#endif

}

uint64_t Kangaroo::FTell(FILE* stream) {

#ifdef WIN64
  return (uint64_t)_ftelli64(stream);
#else
  return (uint64_t)ftello(stream);
#endif

}

bool Kangaroo::IsEmpty(std::string fileName) {

  FILE *pFile = fopen(fileName.c_str(),"r");
  if(pFile==NULL) {
    ::printf("OpenPart: Cannot open %s for reading\n",fileName.c_str());
    ::printf("%s\n",::strerror(errno));
    ::exit(0);
  }
  fseek(pFile,0,SEEK_END);
  uint32_t size = ftell(pFile);
  fclose(pFile);
  return size==0;

}

int Kangaroo::IsDir(string dirName) {

  bool isDir = 0;

#ifdef WIN64

  WIN32_FIND_DATA ffd;
  HANDLE hFind;

  hFind = FindFirstFile(dirName.c_str(),&ffd);
  if(hFind == INVALID_HANDLE_VALUE) {
    ::printf("%s not found\n",dirName.c_str());
    return -1;
  }
  isDir = (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  FindClose(hFind);

#else

  struct stat buffer;
  if(stat(dirName.c_str(),&buffer) != 0) {
    ::printf("%s not found\n",dirName.c_str());
    return -1;
  }
  isDir = (buffer.st_mode & S_IFDIR) != 0;

#endif

  return isDir;

}

FILE *Kangaroo::ReadHeader(std::string fileName, uint32_t *version, uint32_t type) {

  FILE *f = fopen(fileName.c_str(),"rb");
  if(f == NULL) {
    ::printf("ReadHeader: Cannot open %s for reading\n",fileName.c_str());
    ::printf("%s\n",::strerror(errno));
    return NULL;
  }
  uint32_t head;
  uint32_t versionF;

  // Read header
  if(::fread(&head,sizeof(uint32_t),1,f) != 1) {
    ::printf("ReadHeader: Cannot read from %s\n",fileName.c_str());
    if(::feof(f)) {
      ::printf("Empty file\n");
    } else {
      ::printf("%s\n",::strerror(errno));
    }
    ::fclose(f);
    return NULL;
  }

  ::fread(&versionF,sizeof(uint32_t),1,f);
  if(version) *version = versionF;

  if(head!=type) {
    if(head==HEADK) {
      fread(&nbLoadedWalk,sizeof(uint64_t),1,f);
      ::printf("ReadHeader: %s is a kangaroo only file [2^%.2f kangaroos]\n",fileName.c_str(),log2((double)nbLoadedWalk));
    } if(head == HEADKS) {
      fread(&nbLoadedWalk,sizeof(uint64_t),1,f);
      ::printf("ReadHeader: %s is a compressed kangaroo only file [2^%.2f kangaroos]\n",fileName.c_str(),log2((double)nbLoadedWalk));
    } else if(head==HEADW) {
      ::printf("ReadHeader: %s is a work file, kangaroo only file expected\n",fileName.c_str());
    } else {
      ::printf("ReadHeader: %s Not a work file\n",fileName.c_str());
    }
    ::fclose(f);
    return NULL;
  }

  return f;

}

bool Kangaroo::LoadWork(string &fileName) {

  double t0 = Timer::get_tick();

  ::printf("Loading: %s\n",fileName.c_str());

  if(!clientMode) {

    fRead = ReadHeader(fileName,NULL,HEADW);
    if(fRead == NULL)
      return false;

    keysToSearch.clear();
    Point key;

    // Read global param
    uint32_t dp;
    ::fread(&dp,sizeof(uint32_t),1,fRead);
    if(initDPSize < 0) initDPSize = dp;
    ::fread(&rangeStart.bits64,32,1,fRead); rangeStart.bits64[4] = 0;
    ::fread(&rangeEnd.bits64,32,1,fRead); rangeEnd.bits64[4] = 0;
    ::fread(&key.x.bits64,32,1,fRead); key.x.bits64[4] = 0;
    ::fread(&key.y.bits64,32,1,fRead); key.y.bits64[4] = 0;
    ::fread(&offsetCount,sizeof(uint64_t),1,fRead);
    ::fread(&offsetTime,sizeof(double),1,fRead);

    key.z.SetInt32(1);
    if(!secp->EC(key)) {
      ::printf("LoadWork: key does not lie on elliptic curve\n");
      return false;
    }

    keysToSearch.push_back(key);

    ::printf("Start:%s\n",rangeStart.GetBase16().c_str());
    ::printf("Stop :%s\n",rangeEnd.GetBase16().c_str());
    ::printf("Keys :%d\n",(int)keysToSearch.size());

    // Read hashTable
    hashTable.LoadTable(fRead);

  } else {

    // In client mode, config come from the server, file has only kangaroo
    fRead = ReadHeader(fileName,NULL,HEADK);
    if(fRead == NULL)
      return false;

  }

  // Read number of walk
  fread(&nbLoadedWalk,sizeof(uint64_t),1,fRead);

  double t1 = Timer::get_tick();

  ::printf("LoadWork: [HashTable %s] [%s]\n",hashTable.GetSizeInfo().c_str(),GetTimeStr(t1 - t0).c_str());

  return true;
}

// ----------------------------------------------------------------------------

void Kangaroo::FetchWalks(uint64_t nbWalk,Int *x,Int *y,Int *d) {

  // Read Kangaroos
  int64_t n = 0;

  ::printf("Fetch kangaroos: %.0f\n",(double)nbWalk);

  for(n = 0; n < (int64_t)nbWalk && nbLoadedWalk>0; n++) {
    ::fread(&x[n].bits64,32,1,fRead); x[n].bits64[4] = 0;
    ::fread(&y[n].bits64,32,1,fRead); y[n].bits64[4] = 0;
    ::fread(&d[n].bits64,32,1,fRead); d[n].bits64[4] = 0;
    nbLoadedWalk--;
  }

  if(n<(int64_t)nbWalk) {
    int64_t empty = nbWalk - n;
    // Fill empty kanagaroo
    CreateHerd((int)empty,&(x[n]),&(y[n]),&(d[n]),TAME);
  }

}

void Kangaroo::FetchWalks(uint64_t nbWalk,std::vector<int256_t>& kangs,Int* x,Int* y,Int* d) {

  uint64_t n = 0;

  uint64_t avail = (nbWalk<kangs.size())?nbWalk:kangs.size();

  if(avail > 0) {

    vector<Int> dists;
    vector<Point> Sp;
    dists.reserve(avail);
    Sp.reserve(avail);
    Point Z;
    Z.Clear();

    for(n = 0; n < avail; n++) {

      Int dist;
      HashTable::CalcDist(&kangs[n],&dist);
      dists.push_back(dist);

    }

    vector<Point> P = secp->ComputePublicKeys(dists);

    for(n = 0; n < avail; n++) {

      if(n % 2 == TAME) {
        Sp.push_back(Z);
      }
      else {
        Sp.push_back(keyToSearch);
      }

    }

    vector<Point> S = secp->AddDirect(Sp,P);

    for(n = 0; n < avail; n++) {
      x[n].Set(&S[n].x);
      y[n].Set(&S[n].y);
      d[n].Set(&dists[n]);
      nbLoadedWalk--;
    }

    kangs.erase(kangs.begin(),kangs.begin() + avail);
  }

  if(avail < nbWalk) {
    int64_t empty = nbWalk - avail;
    // Fill empty kanagaroo
    CreateHerd((int)empty,&(x[n]),&(y[n]),&(d[n]),TAME);
  }

}

void Kangaroo::FectchKangaroos(TH_PARAM *threads) {

  double sFetch = Timer::get_tick();

  // From server
  vector<int256_t> kangs;
  if(saveKangarooByServer) {
    ::printf("FectchKangaroosFromServer");
    if(!GetKangaroosFromServer(workFile,kangs))
      ::exit(0);
    ::printf("Done\n");
    nbLoadedWalk = kangs.size();
  }


  // Fetch input kangaroo from file (if any)
  if(nbLoadedWalk>0) {

    ::printf("Restoring");

    uint64_t nbSaved = nbLoadedWalk;
    uint64_t created = 0;

    // Fetch loaded walk
    for(int i = 0; i < nbCPUThread; i++) {
      threads[i].px = new Int[CPU_GRP_SIZE];
      threads[i].py = new Int[CPU_GRP_SIZE];
      threads[i].distance = new Int[CPU_GRP_SIZE];
      if(!saveKangarooByServer)
        FetchWalks(CPU_GRP_SIZE,threads[i].px,threads[i].py,threads[i].distance);
      else
        FetchWalks(CPU_GRP_SIZE,kangs,threads[i].px,threads[i].py,threads[i].distance);
    }

#ifdef WITHGPU
    for(int i = 0; i < nbGPUThread; i++) {
      ::printf(".");
      int id = nbCPUThread + i;
      uint64_t n = threads[id].nbKangaroo;
      threads[id].px = new Int[n];
      threads[id].py = new Int[n];
      threads[id].distance = new Int[n];
      if(!saveKangarooByServer)
          FetchWalks(n,
            threads[id].px,
            threads[id].py,
            threads[id].distance);
      else
          FetchWalks(n,kangs,
            threads[id].px,
            threads[id].py,
            threads[id].distance);
    }
#endif

    ::printf("Done\n");

    double eFetch = Timer::get_tick();

    if(nbLoadedWalk != 0) {
      ::printf("FectchKangaroos: Warning %.0f unhandled kangaroos !\n",(double)nbLoadedWalk);
    }

    if(nbSaved<totalRW)
      created = totalRW - nbSaved;

    ::printf("FectchKangaroos: [2^%.2f kangaroos loaded] [%.0f created] [%s]\n",log2((double)nbSaved),(double)created,GetTimeStr(eFetch - sFetch).c_str());

  }

  // Close input file
  if(fRead) fclose(fRead);

}


// ----------------------------------------------------------------------------
bool Kangaroo::SaveHeader(string fileName,FILE* f,int type,uint64_t totalCount,double totalTime) {

  // Header
  uint32_t head = type;
  uint32_t version = 0;
  if(::fwrite(&head,sizeof(uint32_t),1,f) != 1) {
    ::printf("SaveHeader: Cannot write to %s\n",fileName.c_str());
    ::printf("%s\n",::strerror(errno));
    return false;
  }
  ::fwrite(&version,sizeof(uint32_t),1,f);

  if(type==HEADW) {

    // Save global param
    ::fwrite(&dpSize,sizeof(uint32_t),1,f);
    ::fwrite(&rangeStart.bits64,32,1,f);
    ::fwrite(&rangeEnd.bits64,32,1,f);
    ::fwrite(&keysToSearch[keyIdx].x.bits64,32,1,f);
    ::fwrite(&keysToSearch[keyIdx].y.bits64,32,1,f);
    ::fwrite(&totalCount,sizeof(uint64_t),1,f);
    ::fwrite(&totalTime,sizeof(double),1,f);

  }

  return true;
}

void  Kangaroo::SaveWork(string fileName,FILE *f,int type,uint64_t totalCount,double totalTime) {

  ::printf("\nSaveWork: %s",fileName.c_str());

  // Header
  if(!SaveHeader(fileName,f,type,totalCount,totalTime))
    return;

  // Save hash table
  hashTable.SaveTable(f);

}

uint64_t Kangaroo::SaveWorkTxt(const std::string &fileName,uint64_t totalCount,double totalTime,TH_PARAM *threads,int nbThread,
                       uint64_t totalWalk,bool includeKangaroo) {

  ::printf("\nSaveWorkTxt: %s",fileName.c_str());

  std::ofstream out(fileName);
  if(!out.is_open()) {
    ::printf("\nSaveWorkTxt: Cannot open %s for writing\n",fileName.c_str());
    ::printf("%s\n",::strerror(errno));
    return 0;
  }

  auto int256ToHex = [](const int256_t &v) {
    Int tmp;
    HashTable::toInt(const_cast<int256_t*>(&v), &tmp);
    return tmp.GetBase16();
  };

  out << "VERSION 0\n";
  out << "DP_BITS " << dpSize << "\n";
  out << "START " << rangeStart.GetBase16() << "\n";
  out << "STOP " << rangeEnd.GetBase16() << "\n";
  out << "KEYX " << keysToSearch[keyIdx].x.GetBase16() << "\n";
  out << "KEYY " << keysToSearch[keyIdx].y.GetBase16() << "\n";
  out << "COUNT " << totalCount << "\n";
  out << std::setprecision(17) << "TIME " << totalTime << "\n";
  out << "HASH_SIZE " << HASH_SIZE << "\n";

  for(uint32_t h = 0; h < HASH_SIZE; h++) {
    out << "BUCKET " << h << ' ' << hashTable.E[h].nbItem << ' ' << hashTable.E[h].maxItem << "\n";
    for(uint32_t i = 0; i < hashTable.E[h].nbItem; i++) {
      ENTRY *item = hashTable.E[h].items[i];
      out << "ITEM " << int256ToHex(item->x) << ' ' << int256ToHex(item->d) << ' ' << item->kType << "\n";
    }
  }

  uint64_t kangarooCount = includeKangaroo ? totalWalk : 0;
  out << "KANGAROOS " << kangarooCount << "\n";

  if(includeKangaroo) {
    for(int i = 0; i < nbThread; i++) {
      for(uint64_t n = 0; n < threads[i].nbKangaroo; n++) {
        out << "K "
            << threads[i].px[n].GetBase16() << ' '
            << threads[i].py[n].GetBase16() << ' '
            << threads[i].distance[n].GetBase16() << "\n";
      }
    }
  }

  out.flush();
  std::streampos pos = out.tellp();
  if(pos < 0) {
    return 0;
  }
  return static_cast<uint64_t>(pos);

}

void Kangaroo::SaveServerWork() {

  WaitForAsyncSave();

  saveRequest = true;

  double t0 = Timer::get_tick();

  string fileName = workFile;
  if(splitWorkfile)
    fileName = workFile + "_" + Timer::getTS();

  FILE *f = fopen(fileName.c_str(),"wb");
  if(f == NULL) {
    ::printf("\nSaveWork: Cannot open %s for writing\n",fileName.c_str());
    ::printf("%s\n",::strerror(errno));
    saveRequest = false;
    return;
  }

  SaveWork(fileName,f,HEADW,0,0);

  uint64_t totalWalk = 0;
  ::fwrite(&totalWalk,sizeof(uint64_t),1,f);

  uint64_t size = FTell(f);
  fclose(f);

  if(splitWorkfile)
    hashTable.Reset();

  double t1 = Timer::get_tick();

  char *ctimeBuff;
  time_t now = time(NULL);
  ctimeBuff = ctime(&now);
  ::printf("done [%.1f MB] [%s] %s",(double)size / (1024.0*1024.0),GetTimeStr(t1 - t0).c_str(),ctimeBuff);

  saveRequest = false;

}

static void WriteHashTableSnapshot(FILE* f,const HashTableSnapshot& snapshot) {
  for(uint32_t h = 0; h < HASH_SIZE; h++) {
    fwrite(&snapshot.bucketSizes[h],sizeof(uint32_t),1,f);
    fwrite(&snapshot.bucketMax[h],sizeof(uint32_t),1,f);
    uint64_t offset = snapshot.bucketOffsets[h];
    for(uint32_t i = 0; i < snapshot.bucketSizes[h]; i++) {
      const auto &entry = snapshot.entries[offset + i];
      fwrite(&(entry.x),32,1,f);
      fwrite(&(entry.d),32,1,f);
      fwrite(&(entry.kType),4,1,f);
    }
  }
}

uint64_t Kangaroo::SaveWorkTxtSnapshot(AsyncSavePayload &payload) {

  ::printf("\nSaveWorkTxt: %s",payload.textFileName.c_str());

  std::ofstream out(payload.textFileName);
  if(!out.is_open()) {
    ::printf("\nSaveWorkTxt: Cannot open %s for writing\n",payload.textFileName.c_str());
    ::printf("%s\n",::strerror(errno));
    return 0;
  }

  auto int256ToHex = [](const int256_t &v) {
    Int tmp;
    HashTable::toInt(const_cast<int256_t*>(&v), &tmp);
    return tmp.GetBase16();
  };

  out << "VERSION 0\n";
  out << "DP_BITS " << payload.dpBits << "\n";
  out << "START " << payload.rangeStartHex << "\n";
  out << "STOP " << payload.rangeEndHex << "\n";
  out << "KEYX " << payload.keyXHex << "\n";
  out << "KEYY " << payload.keyYHex << "\n";
  out << "COUNT " << payload.totalCount << "\n";
  out << std::setprecision(17) << "TIME " << payload.totalTime << "\n";
  out << "HASH_SIZE " << HASH_SIZE << "\n";

  size_t entryIdx = 0;
  for(uint32_t h = 0; h < HASH_SIZE; h++) {
    out << "BUCKET " << h << ' ' << payload.tableSnapshot.bucketSizes[h] << ' ' << payload.tableSnapshot.bucketMax[h] << "\n";
    entryIdx = payload.tableSnapshot.bucketOffsets[h];
    for(uint32_t i = 0; i < payload.tableSnapshot.bucketSizes[h]; i++) {
      const auto &item = payload.tableSnapshot.entries[entryIdx + i];
      out << "ITEM " << int256ToHex(item.x) << ' ' << int256ToHex(item.d) << ' ' << item.kType << "\n";
    }
  }

  uint64_t kangarooCount = payload.saveKangarooText ? payload.textKangarooCount : 0;
  out << "KANGAROOS " << kangarooCount << "\n";

  if(payload.saveKangarooText) {
    for(size_t i = 0; i < payload.kangarooX.size(); i++) {
      out << "K "
          << payload.kangarooX[i].GetBase16() << ' '
          << payload.kangarooY[i].GetBase16() << ' '
          << payload.kangarooD[i].GetBase16() << "\n";
    }
  }

  out.flush();
  std::streampos pos = out.tellp();
  if(pos < 0) {
    return 0;
  }
  return static_cast<uint64_t>(pos);

}

void Kangaroo::WaitForAsyncSave() {
  std::thread local;
  {
    std::lock_guard<std::mutex> guard(asyncSaveThreadMutex);
    local = std::move(asyncSaveThread);
  }
  if(local.joinable()) {
    local.join();
  }
  asyncSaveRunning = false;
}

void Kangaroo::RunAsyncSave(std::shared_ptr<AsyncSavePayload> payload) {

  uint64_t size = 0;
  uint64_t textSize = 0;

  if(payload->needServerSend) {

    ::printf("\nSaveWork (Kangaroo->Server): %s",payload->fileName.c_str());
    SendKangaroosToServer(payload->fileName,payload->kangaroosForServer);
    size = payload->kangaroosForServer.size() * 32 + 32;

  } else if(payload->hasBinaryTarget) {

    FILE* f = fopen(payload->fileName.c_str(),"wb");
    if(f == NULL) {
      ::printf("\nSaveWork: Cannot open %s for writing\n",payload->fileName.c_str());
      ::printf("%s\n",::strerror(errno));
    } else {
      SaveHeader(payload->fileName,f,payload->headType,payload->totalCount,payload->totalTime);
      ::printf("\nSaveWork: %s",payload->fileName.c_str());
      WriteHashTableSnapshot(f,payload->tableSnapshot);

      ::fwrite(&payload->totalWalk,sizeof(uint64_t),1,f);

      if(payload->saveKangaroo) {
        uint64_t point = payload->totalWalk / 16;
        uint64_t pointPrint = 0;

        for(size_t i = 0; i < payload->kangarooX.size(); i++) {
          ::fwrite(&payload->kangarooX[i].bits64,32,1,f);
          ::fwrite(&payload->kangarooY[i].bits64,32,1,f);
          ::fwrite(&payload->kangarooD[i].bits64,32,1,f);
          pointPrint++;
          if(pointPrint>point) {
            ::printf(".");
            pointPrint = 0;
          }
        }
      }

      size = FTell(f);
      fclose(f);
    }

  }

  if(payload->hasTextTarget)
    textSize = SaveWorkTxtSnapshot(*payload);

  double t1 = Timer::get_tick();

  char *ctimeBuff;
  time_t now = time(NULL);
  ctimeBuff = ctime(&now);
  uint64_t reportedSize = (size > 0) ? size : textSize;
  ::printf("done [%.1f MB] [%s] %s",(double)reportedSize/(1024.0*1024.0),GetTimeStr(t1 - payload->startTick).c_str(),ctimeBuff);

  {
    std::lock_guard<std::mutex> guard(asyncSaveThreadMutex);
    asyncSaveRunning = false;
  }

}

void Kangaroo::SaveWork(uint64_t totalCount,double totalTime,TH_PARAM *threads,int nbThread) {

  if(asyncSaveRunning.load()) {
    ::printf("\nSaveWork: async flush still running, skipping new snapshot\n");
    return;
  }

  WaitForAsyncSave();

  LOCK(saveMutex);

  double t0 = Timer::get_tick();

  saveRequest = true;
  while(!isWaiting(threads) && isAlive(threads) && !endOfSearch) {
    Timer::SleepMillis(10);
  }

  string ts;
  if(splitWorkfile && (workFile.length() > 0 || workTextFile.length() > 0))
    ts = "_" + Timer::getTS();

  string fileName = workFile;
  if(fileName.length() > 0)
    fileName += ts;

  string textFileName = workTextFile;
  if(textFileName.length() > 0)
    textFileName += ts;

  bool hasBinaryTarget = (!saveKangarooByServer) && (fileName.length() > 0);
  bool needServerSend = clientMode && saveKangarooByServer;
  bool hasTextTarget = textFileName.length() > 0;

  uint64_t actualKangarooCount = 0;
  if(saveKangaroo || saveKangarooText || saveKangarooByServer) {
    for(int i = 0; i < nbThread; i++)
      actualKangarooCount += threads[i].nbKangaroo;
  }

  auto payload = std::make_shared<AsyncSavePayload>();
  payload->fileName = fileName;
  payload->textFileName = textFileName;
  payload->hasBinaryTarget = hasBinaryTarget;
  payload->needServerSend = needServerSend;
  payload->hasTextTarget = hasTextTarget;
  payload->saveKangaroo = saveKangaroo;
  payload->saveKangarooText = saveKangarooText;
  payload->splitWorkfile = splitWorkfile;
  payload->totalWalk = saveKangaroo ? actualKangarooCount : 0;
  payload->textKangarooCount = saveKangarooText ? actualKangarooCount : 0;
  payload->dpBits = dpSize;
  payload->rangeStartHex = rangeStart.GetBase16();
  payload->rangeEndHex = rangeEnd.GetBase16();
  payload->keyXHex = keysToSearch[keyIdx].x.GetBase16();
  payload->keyYHex = keysToSearch[keyIdx].y.GetBase16();
  payload->totalCount = totalCount;
  payload->totalTime = totalTime;
  payload->startTick = t0;
  payload->headType = clientMode ? HEADK : HEADW;

  payload->tableSnapshot.bucketSizes.resize(HASH_SIZE);
  payload->tableSnapshot.bucketMax.resize(HASH_SIZE);
  payload->tableSnapshot.bucketOffsets.resize(HASH_SIZE);
  payload->tableSnapshot.entries.reserve(hashTable.GetNbItem());

  uint64_t entryOffset = 0;
  for(uint32_t h = 0; h < HASH_SIZE; h++) {
    payload->tableSnapshot.bucketOffsets[h] = entryOffset;
    payload->tableSnapshot.bucketSizes[h] = hashTable.E[h].nbItem;
    payload->tableSnapshot.bucketMax[h] = hashTable.E[h].maxItem;
    for(uint32_t i = 0; i < hashTable.E[h].nbItem; i++) {
      HashEntrySnapshot snap;
      snap.x = hashTable.E[h].items[i]->x;
      snap.d = hashTable.E[h].items[i]->d;
      snap.kType = hashTable.E[h].items[i]->kType;
      payload->tableSnapshot.entries.push_back(snap);
    }
    entryOffset += hashTable.E[h].nbItem;
  }

  if(saveKangaroo || saveKangarooText || saveKangarooByServer) {
    payload->kangarooX.reserve(actualKangarooCount);
    payload->kangarooY.reserve(actualKangarooCount);
    payload->kangarooD.reserve(actualKangarooCount);

    if(needServerSend)
      payload->kangaroosForServer.reserve(actualKangarooCount);

    for(int i = 0; i < nbThread; i++) {
      for(uint64_t n = 0; n < threads[i].nbKangaroo; n++) {
        payload->kangarooX.push_back(threads[i].px[n]);
        payload->kangarooY.push_back(threads[i].py[n]);
        payload->kangarooD.push_back(threads[i].distance[n]);

        if(needServerSend) {
          int256_t X;
          int256_t D;
          HashTable::Convert(&threads[i].px[n],&threads[i].distance[n],&X,&D);
          payload->kangaroosForServer.push_back(D);
        }
      }
    }
  }

  saveRequest = false;

  if(splitWorkfile && (hasBinaryTarget || hasTextTarget))
    hashTable.Reset();

  UNLOCK(saveMutex);

  if(!hasBinaryTarget && !hasTextTarget && !needServerSend)
    return;

  ::printf("\nSaveWork: captured snapshot for async flush\n");

  {
    std::lock_guard<std::mutex> guard(asyncSaveThreadMutex);
    asyncSaveRunning = true;
    asyncSaveThread = std::thread(&Kangaroo::RunAsyncSave,this,payload);
  }

}

void Kangaroo::WorkInfo(std::string &fName) {

  int isDir = IsDir(fName);
  if(isDir<0)
    return;

  string fileName = fName;
  if(isDir)
    fileName = fName + "/header";

  ::printf("Loading: %s\n",fileName.c_str());

  uint32_t version;
  FILE *f1 = ReadHeader(fileName,&version,HEADW);
  if(f1 == NULL)
    return;

#ifndef WIN64
#if defined(POSIX_FADV_RANDOM) && defined(POSIX_FADV_NOREUSE)
  int fd = fileno(f1);
  posix_fadvise(fd,0,0,POSIX_FADV_RANDOM|POSIX_FADV_NOREUSE);
#endif
#endif

  uint32_t dp1;
  Point k1;
  uint64_t count1;
  double time1;
  Int RS1;
  Int RE1;

  // Read global param
  ::fread(&dp1,sizeof(uint32_t),1,f1);
  ::fread(&RS1.bits64,32,1,f1); RS1.bits64[4] = 0;
  ::fread(&RE1.bits64,32,1,f1); RE1.bits64[4] = 0;
  ::fread(&k1.x.bits64,32,1,f1); k1.x.bits64[4] = 0;
  ::fread(&k1.y.bits64,32,1,f1); k1.y.bits64[4] = 0;
  ::fread(&count1,sizeof(uint64_t),1,f1);
  ::fread(&time1,sizeof(double),1,f1);

  k1.z.SetInt32(1);
  if(!secp->EC(k1)) {
    ::printf("WorkInfo: key1 does not lie on elliptic curve\n");
    fclose(f1);
    return;
  }

  // Read hashTable
  if(isDir) {
    for(int i = 0; i < MERGE_PART; i++) {
      FILE* f = OpenPart(fName,"rb",i);
      hashTable.SeekNbItem(f,i * H_PER_PART,(i + 1) * H_PER_PART);
      fclose(f);
    }
  } else {
    hashTable.SeekNbItem(f1);
  }

  ::printf("Version   : %d\n",version);
  ::printf("DP bits   : %d\n",dp1);
  ::printf("Start     : %s\n",RS1.GetBase16().c_str());
  ::printf("Stop      : %s\n",RE1.GetBase16().c_str());
  ::printf("Key       : %s\n",secp->GetPublicKeyHex(true,k1).c_str());
#ifdef WIN64
  ::printf("Count     : %I64d 2^%.3f\n",count1,log2(count1));
#else
  ::printf("Count     : %" PRId64 " 2^%.3f\n",count1,log2(count1));
#endif
  ::printf("Time      : %s\n",GetTimeStr(time1).c_str());
  hashTable.PrintInfo();

  fread(&nbLoadedWalk,sizeof(uint64_t),1,f1);
#ifdef WIN64
  ::printf("Kangaroos : %I64d 2^%.3f\n",nbLoadedWalk,log2(nbLoadedWalk));
#else
  ::printf("Kangaroos : %" PRId64 " 2^%.3f\n",nbLoadedWalk,log2(nbLoadedWalk));
#endif

  fclose(f1);

}
