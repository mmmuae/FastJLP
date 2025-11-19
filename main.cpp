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
#include "Timer.h"
#include "SECPK1/SECP256k1.h"
#include "GPU/GPUEngine.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

#if defined(__SIZEOF_INT128__)
__extension__ typedef unsigned __int128 uint128_ext;
#endif

using namespace std;

#define CHECKARG(opt,n) if(a>=argc-1) {::printf(opt " missing argument #%d\n",n);exit(0);} else {a++;}

static bool is_hex_string(const std::string &s);
static bool is_pubkey_hex(const std::string &s);
static bool dec_to_u256(const std::string &dec, std::array<uint64_t,4> &out_le);
static std::string u256_to_hex64_be(const std::array<uint64_t,4> &v_le);
static bool hex_to_u256(const std::string &hex, std::array<uint64_t,4> &out_le, std::string &out_hex64);
static int  safe_write_all(int fd, const void *buf, size_t len);
static std::string make_ephemeral_config(const std::string &start_hex64,
                                         const std::string &end_hex64,
                                         const std::string &pubkey_hex,
                                         std::string &tmp_path_out);
static void cleanup_cli_config();

// ------------------------------------------------------------------------------------------

void printUsage() {

  printf("Kangaroo-256 [-v] [-t nbThread] [-d dpBit] [gpu] [-check]\n");
  printf("         [-gpuId gpuId1[,gpuId2,...]] [-g g1x,g1y[,g2x,g2y,...]]\n");
  printf("         inFile\n");
  printf(" -v: Print version\n");
  printf(" -gpu: Enable gpu calculation\n");
  printf(" -gpuId gpuId1,gpuId2,...: List of GPU(s) to use, default is 0\n");
  printf(" -g g1x,g1y,g2x,g2y,...: Specify GPU(s) kernel gridsize, default is 2*(MP),2*(Core/MP)\n");
  printf(" -d: Specify number of leading zeros for the DP method (default is auto)\n");
  printf(" -t nbThread: Secify number of thread\n");
  printf(" -w workfile: Specify file to save work into (current processed key only)\n");
  printf(" -i workfile: Specify file to load work from (current processed key only)\n");
  printf(" -wi workInterval: Periodic interval (in seconds) for saving work\n");
  printf(" -ws: Save kangaroos in the work file\n");
  printf(" -wss: Save kangaroos via the server\n");
  printf(" -wsplit: Split work file of server and reset hashtable\n");
  printf(" -wm file1 file2 destfile: Merge work file\n");
  printf(" -wmdir dir destfile: Merge directory of work files\n");
  printf(" -wt timeout: Save work timeout in millisec (default is 3000ms)\n");
  printf(" -winfo file1: Work file info file\n");
  printf(" -wpartcreate name: Create empty partitioned work file (name is a directory)\n");
  printf(" -wcheck worfile: Check workfile integrity\n");
  printf(" -m maxStep: number of operations before give up the search (maxStep*expected operation)\n");
  printf(" -s: Start in server mode\n");
  printf(" -c server_ip: Start in client mode and connect to server server_ip\n");
  printf(" -sp port: Server port, default is 17403\n");
  printf(" -nt timeout: Network timeout in millisec (default is 3000ms)\n");
  printf(" -o fileName: output result to fileName\n");
  printf(" -l: List cuda enabled devices\n");
  printf(" -check: Check GPU kernel vs CPU\n");
  printf(" --start-dec/--end-dec/--pubkey: Provide decimal bounds + pubkey via CLI (temp config)\n");
  printf(" --start-hex/--end-hex/--pubkey: Provide hex bounds + pubkey via CLI (temp config)\n");
  printf(" inFile: intput configuration file\n");
  exit(0);

}

// ------------------------------------------------------------------------------------------

int getInt(string name,char *v) {

  int r;

  try {

    r = std::stoi(string(v));

  } catch(std::invalid_argument&) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

  return r;

}

double getDouble(string name,char *v) {

  double r;

  try {

    r = std::stod(string(v));

  } catch(std::invalid_argument&) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

  return r;

}

// ------------------------------------------------------------------------------------------

void getInts(string name,vector<int> &tokens,const string &text,char sep) {

  size_t start = 0,end = 0;
  tokens.clear();
  int item;

  try {

    while((end = text.find(sep,start)) != string::npos) {
      item = std::stoi(text.substr(start,end - start));
      tokens.push_back(item);
      start = end + 1;
    }

    item = std::stoi(text.substr(start));
    tokens.push_back(item);

  }
  catch(std::invalid_argument &) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

}
// ------------------------------------------------------------------------------------------

// Default params
static int dp = -1;
static int nbCPUThread;
static string configFile = "";
static bool checkFlag = false;
static bool gpuEnable = false;
static vector<int> gpuId = { 0 };
static vector<int> gridSize;
static string workFile = "";
static string checkWorkFile = "";
static string iWorkFile = "";
static uint32_t savePeriod = 60;
static bool saveKangaroo = false;
static bool saveKangarooByServer = false;
static string merge1 = "";
static string merge2 = "";
static string mergeDest = "";
static string mergeDir = "";
static string infoFile = "";
static double maxStep = 0.0;
static int wtimeout = 3000;
static int ntimeout = 3000;
static int port = 17403;
static bool serverMode = false;
static string serverIP = "";
static string outputFile = "";
static bool splitWorkFile = false;

static string cli_start_dec;
static string cli_end_dec;
static string cli_start_hex;
static string cli_end_hex;
static string cli_pubkey_hex;

static string cli_temp_config_path;
static bool cli_cleanup_registered = false;

int main(int argc, char* argv[]) {

#ifdef USE_SYMMETRY
  printf("Kangaroo v" RELEASE " (with symmetry [256 range edition by NotATether])\n");
#else
  printf("Kangaroo v" RELEASE " [256 range edition by NotATether]\n");
#endif

  // Global Init
  Timer::Init();
  rseed(Timer::getSeed32());

  // Init SecpK1
  Secp256K1 *secp = new Secp256K1();
  secp->Init();

  int a = 1;
  nbCPUThread = Timer::getCoreNumber();

  while (a < argc) {

    if(strcmp(argv[a], "-t") == 0) {
      CHECKARG("-t",1);
      nbCPUThread = getInt("nbCPUThread",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-d") == 0) {
      CHECKARG("-d",1);
      dp = getInt("dpSize",argv[a]);
      a++;
    } else if (strcmp(argv[a], "-h") == 0) {
      printUsage();
    } else if(strcmp(argv[a],"-l") == 0) {

#ifdef WITHGPU
      GPUEngine::PrintCudaInfo();
#else
      printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
      exit(0);

    } else if(strcmp(argv[a],"-w") == 0) {
      CHECKARG("-w",1);
      workFile = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-i") == 0) {
      CHECKARG("-i",1);
      iWorkFile = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-wm") == 0) {
      CHECKARG("-wm",1);
      merge1 = string(argv[a]);
      CHECKARG("-wm",2);
      merge2 = string(argv[a]);
      a++;
      if(a<argc) {
        // classic merge
        mergeDest = string(argv[a]);
        a++;
      }
    } else if(strcmp(argv[a],"-wmdir") == 0) {
      CHECKARG("-wmdir",1);
      mergeDir = string(argv[a]);
      CHECKARG("-wmdir",2);
      mergeDest = string(argv[a]);
      a++;
    }  else if(strcmp(argv[a],"-wcheck") == 0) {
      CHECKARG("-wcheck",1);
      checkWorkFile = string(argv[a]);
      a++;
    }  else if(strcmp(argv[a],"-winfo") == 0) {
      CHECKARG("-winfo",1);
      infoFile = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-o") == 0) {
      CHECKARG("-o",1);
      outputFile = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-wi") == 0) {
      CHECKARG("-wi",1);
      savePeriod = getInt("savePeriod",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-wt") == 0) {
      CHECKARG("-wt",1);
      wtimeout = getInt("timeout",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-nt") == 0) {
      CHECKARG("-nt",1);
      ntimeout = getInt("timeout",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-m") == 0) {
      CHECKARG("-m",1);
      maxStep = getDouble("maxStep",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-ws") == 0) {
      a++;
      saveKangaroo = true;
    } else if(strcmp(argv[a],"-wss") == 0) {
      a++;
      saveKangarooByServer = true;
    } else if(strcmp(argv[a],"-wsplit") == 0) {
      a++;
      splitWorkFile = true;
    } else if(strcmp(argv[a],"-wpartcreate") == 0) {
      CHECKARG("-wpartcreate",1);
      workFile = string(argv[a]);
      Kangaroo::CreateEmptyPartWork(workFile);
      exit(0);
    } else if(strcmp(argv[a],"-s") == 0) {
      if (serverIP != "") {
	printf("-s and -c are incompatible\n");
	exit(-1);
      }
      a++;
      serverMode = true;
    } else if(strcmp(argv[a],"-c") == 0) {
      CHECKARG("-c",1);
      if (serverMode) {
        printf("-s and -c are incompatible\n");
	exit(-1);
      }
      serverIP = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-sp") == 0) {
      CHECKARG("-sp",1);
      port = getInt("serverPort",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-gpu") == 0) {
      gpuEnable = true;
      a++;
    } else if(strcmp(argv[a],"-gpuId") == 0) {
      CHECKARG("-gpuId",1);
      getInts("gpuId",gpuId,string(argv[a]),',');
      a++;
    } else if(strcmp(argv[a],"-g") == 0) {
      CHECKARG("-g",1);
      getInts("gridSize",gridSize,string(argv[a]),',');
      a++;
    } else if(strcmp(argv[a],"--start-dec") == 0) {
      CHECKARG("--start-dec",1);
      cli_start_dec = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"--end-dec") == 0) {
      CHECKARG("--end-dec",1);
      cli_end_dec = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"--start-hex") == 0) {
      CHECKARG("--start-hex",1);
      cli_start_hex = string(argv[a]);
      std::transform(cli_start_hex.begin(), cli_start_hex.end(), cli_start_hex.begin(),
                     [](unsigned char c){ return static_cast<char>(std::toupper(c)); });
      a++;
    } else if(strcmp(argv[a],"--end-hex") == 0) {
      CHECKARG("--end-hex",1);
      cli_end_hex = string(argv[a]);
      std::transform(cli_end_hex.begin(), cli_end_hex.end(), cli_end_hex.begin(),
                     [](unsigned char c){ return static_cast<char>(std::toupper(c)); });
      a++;
    } else if(strcmp(argv[a],"--pubkey") == 0) {
      CHECKARG("--pubkey",1);
      cli_pubkey_hex = string(argv[a]);
      std::transform(cli_pubkey_hex.begin(), cli_pubkey_hex.end(), cli_pubkey_hex.begin(),
                     [](unsigned char c){ return static_cast<char>(std::toupper(c)); });
      a++;
    } else if(strcmp(argv[a],"-v") == 0) {
      ::exit(0);
    } else if(strcmp(argv[a],"-check") == 0) {
      checkFlag = true;
      a++;
    } else if(a == argc - 1) {
      configFile = string(argv[a]);
      a++;
    } else {
      printf("Unexpected %s argument\n",argv[a]);
      exit(-1);
    }

  }

  bool using_cli_config = false;
  if(!cli_start_dec.empty() || !cli_end_dec.empty() ||
     !cli_start_hex.empty() || !cli_end_hex.empty() ||
     !cli_pubkey_hex.empty()) {
    const bool have_dec = (!cli_start_dec.empty() || !cli_end_dec.empty());
    const bool have_hex = (!cli_start_hex.empty() || !cli_end_hex.empty());
    if(have_dec && have_hex) {
      printf("Error: do not mix decimal and hexadecimal CLI ranges\n");
      exit(-1);
    }
    if(!have_dec && !have_hex) {
      printf("Error: missing CLI range\n");
      exit(-1);
    }
    if(cli_pubkey_hex.empty()) {
      printf("Error: --pubkey HEX is required with CLI range input\n");
      exit(-1);
    }
    if(!is_pubkey_hex(cli_pubkey_hex)) {
      printf("Error: --pubkey must be a valid compressed/uncompressed hex key\n");
      exit(-1);
    }

    std::string start_hex64;
    std::string end_hex64;
    if(have_dec) {
      std::array<uint64_t,4> start_le = {0,0,0,0};
      std::array<uint64_t,4> end_le = {0,0,0,0};
      if(cli_start_dec.empty() || cli_end_dec.empty()) {
        printf("Error: --start-dec and --end-dec required together\n");
        exit(-1);
      }
      if(!dec_to_u256(cli_start_dec, start_le)) {
        printf("Error: invalid --start-dec value\n");
        exit(-1);
      }
      if(!dec_to_u256(cli_end_dec, end_le)) {
        printf("Error: invalid --end-dec value\n");
        exit(-1);
      }
      bool ok = true;
      for(int i = 3; i >= 0; --i) {
        if(start_le[i] > end_le[i]) { ok = false; break; }
        if(start_le[i] < end_le[i]) { break; }
      }
      if(!ok) {
        printf("Error: start-dec must be <= end-dec\n");
        exit(-1);
      }
      start_hex64 = u256_to_hex64_be(start_le);
      end_hex64   = u256_to_hex64_be(end_le);
    } else {
      std::array<uint64_t,4> start_le = {0,0,0,0};
      std::array<uint64_t,4> end_le = {0,0,0,0};
      if(cli_start_hex.empty() || cli_end_hex.empty()) {
        printf("Error: --start-hex and --end-hex required together\n");
        exit(-1);
      }
      if(!hex_to_u256(cli_start_hex, start_le, start_hex64)) {
        printf("Error: invalid --start-hex value\n");
        exit(-1);
      }
      if(!hex_to_u256(cli_end_hex, end_le, end_hex64)) {
        printf("Error: invalid --end-hex value\n");
        exit(-1);
      }
      bool ok = true;
      for(int i = 3; i >= 0; --i) {
        if(start_le[i] > end_le[i]) { ok = false; break; }
        if(start_le[i] < end_le[i]) { break; }
      }
      if(!ok) {
        printf("Error: start-hex must be <= end-hex\n");
        exit(-1);
      }
    }

    std::string tmp_path;
    std::string cfg = make_ephemeral_config(start_hex64, end_hex64, cli_pubkey_hex, tmp_path);
    if(cfg.empty()) {
      printf("Error: failed to create temporary config\n");
      exit(-1);
    }
    configFile = cfg;
    cli_temp_config_path = tmp_path;
    using_cli_config = true;
    if(!cli_cleanup_registered) {
      atexit(cleanup_cli_config);
      cli_cleanup_registered = true;
    }
  }

  if(gridSize.size() == 0) {
    for(size_t i = 0; i < gpuId.size(); i++) {
      gridSize.push_back(0);
      gridSize.push_back(0);
    }
  } else if(gridSize.size() != gpuId.size() * 2) {
    printf("Invalid gridSize or gpuId argument, must have coherent size\n");
    exit(-1);
  }

  Kangaroo *v = new Kangaroo(secp,dp,gpuEnable,workFile,iWorkFile,savePeriod,saveKangaroo,saveKangarooByServer,
                             maxStep,wtimeout,port,ntimeout,serverIP,outputFile,splitWorkFile);
  if(checkFlag) {
    v->Check(gpuId,gridSize);  
    exit(0);
  } else {
    if(checkWorkFile.length() > 0) {
      v->CheckWorkFile(nbCPUThread,checkWorkFile);
      exit(0);
    } if(infoFile.length()>0) {
      v->WorkInfo(infoFile);
      exit(0);
    } else if(mergeDir.length() > 0) {
      v->MergeDir(mergeDir,mergeDest);
      exit(0);
    } else if(merge1.length()>0) {
      v->MergeWork(merge1,merge2,mergeDest);
      exit(0);
    } if(iWorkFile.length()>0) {
      if( !v->LoadWork(iWorkFile) )
        exit(-1);
    } else if(configFile.length()>0) {
      if( !v->ParseConfigFile(configFile) )
        exit(-1);
    } else {
      if(serverIP.length()==0) {
        ::printf("No input file to process\n");
        exit(-1);
      }
    }
    if(serverMode)
      v->RunServer();
    else
      v->Run(nbCPUThread,gpuId,gridSize);
  }

  if(using_cli_config) {
    cleanup_cli_config();
  }

  return 0;

}

static bool is_hex_string(const std::string &s) {
  if(s.empty()) {
    return false;
  }
  for(unsigned char c : s) {
    if(!std::isxdigit(c)) {
      return false;
    }
  }
  return true;
}

static bool is_pubkey_hex(const std::string &s) {
  if(!(s.size() == 66 || s.size() == 130)) {
    return false;
  }
  if(!is_hex_string(s)) {
    return false;
  }
  if(!(s.rfind("02",0) == 0 || s.rfind("03",0) == 0 || s.rfind("04",0) == 0)) {
    return false;
  }
  return true;
}

static bool dec_to_u256(const std::string &dec, std::array<uint64_t,4> &out_le) {
  out_le = {0,0,0,0};
  if(dec.empty()) {
    return false;
  }
  for(unsigned char c : dec) {
    if(c < '0' || c > '9') {
      return false;
    }
    unsigned int digit = static_cast<unsigned int>(c - '0');
#if defined(__SIZEOF_INT128__)
    uint128_ext carry = digit;
    for(int i = 0; i < 4; ++i) {
      uint128_ext cur = static_cast<uint128_ext>(out_le[i]) * 10u + carry;
      out_le[i] = static_cast<uint64_t>(cur);
      carry = (cur >> 64);
    }
    if(carry != 0) {
      return false;
    }
#else
#error "dec_to_u256 requires compiler support for __int128"
#endif
  }
  return true;
}

static std::string u256_to_hex64_be(const std::array<uint64_t,4> &v_le) {
  char buf[16 * 4 + 1] = {0};
  std::snprintf(buf, sizeof(buf), "%016llX%016llX%016llX%016llX",
                static_cast<unsigned long long>(v_le[3]),
                static_cast<unsigned long long>(v_le[2]),
                static_cast<unsigned long long>(v_le[1]),
                static_cast<unsigned long long>(v_le[0]));
  return std::string(buf);
}

static bool hex_to_u256(const std::string &hex, std::array<uint64_t,4> &out_le, std::string &out_hex64) {
  if(hex.empty()) {
    return false;
  }
  std::string h = hex;
  if(h.size() >= 2 && h[0] == '0' && (h[1] == 'x' || h[1] == 'X')) {
    h = h.substr(2);
  }
  if(h.empty() || h.size() > 64) {
    return false;
  }
  if(!is_hex_string(h)) {
    return false;
  }
  if(h.size() < 64) {
    h = std::string(64 - h.size(), '0') + h;
  }
  std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  out_le = {0,0,0,0};
  for(int limb = 0; limb < 4; ++limb) {
    std::string part = h.substr(limb * 16, 16);
    char *endp = nullptr;
    unsigned long long val = std::strtoull(part.c_str(), &endp, 16);
    if(endp == nullptr || *endp != '\0') {
      return false;
    }
    out_le[3 - limb] = static_cast<uint64_t>(val);
  }
  out_hex64 = h;
  return true;
}

static int safe_write_all(int fd, const void *buf, size_t len) {
#ifndef _WIN32
  const unsigned char *ptr = static_cast<const unsigned char *>(buf);
  size_t written = 0;
  while(written < len) {
    ssize_t res = ::write(fd, ptr + written, len - written);
    if(res < 0) {
      if(errno == EINTR) {
        continue;
      }
      return -1;
    }
    written += static_cast<size_t>(res);
  }
  return 0;
#else
  (void)fd; (void)buf; (void)len;
  errno = ENOSYS;
  return -1;
#endif
}

static std::string make_ephemeral_config(const std::string &start_hex64,
                                         const std::string &end_hex64,
                                         const std::string &pubkey_hex,
                                         std::string &tmp_path_out) {
#ifndef _WIN32
  char tmpl[] = "/tmp/kang_cfg_XXXXXX";
  int fd = ::mkstemp(tmpl);
  if(fd == -1) {
    perror("mkstemp");
    return std::string();
  }
  std::string content;
  content.reserve(64 + 1 + 64 + 1 + pubkey_hex.size() + 1);
  content.append(start_hex64).append("\n");
  content.append(end_hex64).append("\n");
  content.append(pubkey_hex).append("\n");
  if(safe_write_all(fd, content.data(), content.size()) != 0) {
    perror("write");
    ::close(fd);
    ::unlink(tmpl);
    return std::string();
  }
  ::close(fd);
  tmp_path_out.assign(tmpl);
  return std::string(tmpl);
#else
  (void)start_hex64; (void)end_hex64; (void)pubkey_hex; (void)tmp_path_out;
  return std::string();
#endif
}

static void cleanup_cli_config() {
#ifndef _WIN32
  if(!cli_temp_config_path.empty()) {
    ::unlink(cli_temp_config_path.c_str());
    cli_temp_config_path.clear();
  }
#endif
}
