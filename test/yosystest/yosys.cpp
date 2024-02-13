#include <kernel/yosys.h>

//extern "C" int yosysPass(char *command)
//{
//	 Yosys::run_pass(std::string(command));
//}

int main(int argc, char **argv)
{


    Yosys::log_streams.push_back(&std::cout);
    Yosys::log_error_stderr = true;

    Yosys::yosys_setup();
    Yosys::yosys_banner();

    std::string arg = std::string(argv[1]);
    Yosys::run_pass("read_verilog "+arg);

    Yosys::run_pass("dump");
    Yosys::run_pass("proc; flatten; opt -full; synth -noabc;  abc -exe \"/opt/yosys/yosys-abc\" -g AND,OR,XOR;");
//    Yosys::run_pass("proc; flatten; opt -full; -lut abc -exe /opt/yosys -g AND,OR,XOR;");
    //Yosys::run_pass("proc; flatten; opt -full");
    Yosys::run_pass("dump");
    Yosys::run_pass("torder -stop * P*;");
    Yosys::run_pass("ltp");

    Yosys::run_pass("write_blif example.blif");

    Yosys::yosys_shutdown();
    return 0;
}
