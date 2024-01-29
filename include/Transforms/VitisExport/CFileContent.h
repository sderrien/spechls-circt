//
// Created by Steven on 21/01/2024.
//

#ifndef SPECHLS_DIALECT_CFILECONTENT_H
#define SPECHLS_DIALECT_CFILECONTENT_H

#include "mlir/IR/BuiltinOps.h"

#include <algorithm>
#include <iostream>
#include <iosfwd>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

using namespace std;
using namespace mlir;

struct CFileContent {

private:

  std::map<std::string, std::string> opToId;
  std::map<std::string, std::string> valueToId;

  u_int32_t id=0;
  u_int32_t vid=0;
  string path;
  string name;
  vector<string> includes;
  vector<string> declarations;
  vector<string> init;
  vector<string> syncUpdate;
  vector<string> combUpdate;

public:

  CFileContent(string path, string filename) {     // Constructor
    this->name =filename;
    this->path=path;
  }

  bool save();
  string getOpId(mlir::Operation *op) ;
  string getValueId(mlir::Value *v) ;

  void appendIncludesUpdate(string line);
  void appendDeclarations(string line);
  void appendCombUpdate(string line);
  void appendSyncUpdate(string line);
  void appendInitUpdate(string line);

};

#endif // SPECHLS_DIALECT_CFILECONTENT_H
