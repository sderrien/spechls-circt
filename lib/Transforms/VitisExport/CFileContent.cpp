//
// Created by Steven on 21/01/2024.
//

#include "Transforms/VitisExport/CFileContent.h"

using namespace std;
using namespace mlir;

bool CFileContent::save() {
  ofstream oFile;
  oFile.open(path+"/"+name+".cpp");

  for (auto &i : includes)
    oFile << i << "\n";

  for (auto &d : declarations)
    oFile << "\t" << d << "\n";

  oFile << "\tbool exit;\n";
  for (auto &i : init)
    oFile << "\t" << i << "\n";

  oFile << "\tdo {\n";
  oFile << "\t\t // Combinational update\n\n";
  for (auto &c : combUpdate)
    oFile << "\t\t" << c << "\n";

  oFile << "\t\t // Synchronous update\n\n";
  for (auto &s : syncUpdate)
    oFile << "\t\t" << s << "\n";

  oFile << "\t} while (!exit);\n";
  oFile << "} \n";
  oFile.close();
  return 0;
}
inline template <typename T>
string op2str(T *v) {
  std::string s;
  llvm::raw_string_ostream r(s);
  v->print(r);
  return r.str();
}

string CFileContent::getOpId(mlir::Operation *p) {
  auto key = op2str(p);
  auto it = opToId.find(key);
  string res = "";
  if (it==opToId.end()) {
    res = "op_"+to_string(id);
    id = id +1;
    opToId[key] = res;
  } else {
    res = it->second;
  }
  return res;
}

string CFileContent::getValueId(mlir::Value *p) {
    //llvm::outs() << "searching for "<<  p<< " : " << *p << "\n";
    auto key  = op2str(p);
    auto it = valueToId.find(key);
    string res = "";
    if (it==valueToId.end()) {
      res = "v_"+to_string(vid);
      vid = vid +1;
      valueToId[key] = res;
      //llvm::outs() << "\t-adding "<< res << "->" << *p << "\n";
    } else {
      //llvm::outs() << "\t-found "<< it->second << " -> " << (it->first) << "\n";
      res = it->second;
    }
    llvm::outs() << "\t-found "<< res << "\n";
    return res;


}

void CFileContent::appendIncludesUpdate(string line) { includes.push_back(line);}
void CFileContent::appendDeclarations(string line) { declarations.push_back(line);}
void CFileContent::appendCombUpdate(string line) { combUpdate.push_back(line);}
void CFileContent::appendSyncUpdate(string line) { syncUpdate.push_back(line);}
void CFileContent::appendInitUpdate(string line) { init.push_back(line);}



