

#include "llvm/Support/InitLLVM.h"
#include "GecosAPI/gecosapi.h"
#include <kernel/yosys.h>

char* readFile(const char *filename) {
  // Open the file
  FILE *file = fopen(filename, "rb");
  if (!file) {
    perror("Failed to open file");
    return NULL;
  }

  // Determine the file size
  fseek(file, 0, SEEK_END);
  long fileSize = ftell(file);
  fseek(file, 0, SEEK_SET);

  if (fileSize < 0) {
    perror("Failed to determine file size");
    fclose(file);
    return NULL;
  }

  // Allocate memory for the buffer
  char *buffer = (char*)malloc(fileSize + 1); // +1 for the null terminator
  if (!buffer) {
    perror("Failed to allocate memory");
    fclose(file);
    return NULL;
  }

  // Read the file content into the buffer
  size_t bytesRead = fread(buffer, 1, fileSize, file);
  if (bytesRead != fileSize) {
    perror("Failed to read file");
    free(buffer);
    fclose(file);
    return NULL;
  }

  // Null-terminate the buffer
  buffer[fileSize] = '\0';

  // Close the file
  fclose(file);


  return buffer;
}

int main(int argc, char **argv)
#define USE_CIRCT
#ifdef USE_CIRCT


{
  llvm::InitLLVM y(argc, argv);
  llvm::outs() << "Hello\n";
  llvm::errs() << "Hello\n";
  mlir::DialectRegistry registry;
  // registerAllDialects(registry);
  registry.insert<circt::comb::CombDialect, circt::seq::SeqDialect,
                  circt::hw::HWDialect, circt::sv::SVDialect>();
  registry.insert<SpecHLS::SpecHLSDialect>();
  registry.insert<SpecHLS::ScheduleDialectDialect>();


  SpecHLS::registerYosysOptimizerPass();


  for (int k=1; k<argc;k++) {
    char *content = readFile(argv[k]);
    if (content!=NULL) {
      MlirModule m = parseMLIR(content);
      yosysOptimizer(m);
      free(content);


      //destroyMLIR(m);
    }
  }

  llvm::outs() << "Yosys is done\n";
#else

    Yosys::log_streams.push_back(&std::cout);
Yosys::log_error_stderr = true;

Yosys::yosys_setup();
Yosys::yosys_banner();

    std::string abcPath = "/usr/local/bin/yosys-abc";

  for (int k=1; k<argc;k++) {
    std::cout << "Processing file " << argv[k] << "\n";
    std::string arg = std::string(argv[k]);
    Yosys::run_pass("read_verilog "+arg);


    Yosys::run_pass("proc; flatten;   ");
    Yosys::run_pass("opt -full;   ");
    //    #ifndef USE_YOSYS_ABC
    Yosys::run_pass("synth -noabc ;  ");
    //    #else
    Yosys::run_pass("abc -exe " + abcPath + " -g AND,OR ;");
    //    #endif
    Yosys::run_pass("hierarchy -generate * o:Y i:*; opt; opt_clean -purge ;");
    Yosys::run_pass("clean -purge ;");
    auto stop = std::chrono::high_resolution_clock::now();
    Yosys::run_pass("torder -stop * P*;");
    Yosys::run_pass("write_verilog " + arg + "_yosys.sv ;");

    Yosys::log_streams.clear();
    std::stringstream cellOrder;

    //auto topologicalOrder = getTopologicalOrder(cellOrder);
    Yosys::RTLIL::Design *design = Yosys::yosys_get_design();
    std::cout << Yosys done " << argv[k] << "\n";

    //llvm::errs() << "Yosys is done\n";

//      char *content = readFile(argv[k]);
//      MlirModule m= parseMLIR(content);
//      yosysOptimizer(m);
//      free(content);
//      std::cout << "Yosys done\n";
//      destroyMLIR(m);
  }

#endif

    return 0;
}
