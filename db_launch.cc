#include <iostream>
#include <sstream>
#include <string>

#include "legion.h"

using namespace Legion;

enum TaskID {
  DISPATCH_TASK_ID,
  GET_TASK_ID,
  SET_TASK_ID,
};

void dispatch_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime) {
    std::string line;
    std::string command;
    uint64_t address;

    std::cout << "> ";
    std::getline(std::cin, line);
    std::stringstream iss(line);
    iss >> command >> address;
    if (command == "get") {
        std::cout << "Reading address" << address << std::endl;
        TaskLauncher launcher(GET_TASK_ID, TaskArgument(&address, sizeof(address)));
        Future get_future = runtime->execute_task(ctx, launcher);
        std::cout << "Value is: " << get_future.get_result<int>() << std::endl;
    } else if (command == "set") {
        int64_t value;
        iss >> value;
        std::cout << "Setting address " << address << " to " << value << std::endl;
    } else {
        std::cout << "Unrecognized command: " << command << std::endl;
        std::cout << "Allowed commands:" << std::endl;
        std::cout << "\tget <address>" << std::endl;
        std::cout << "\tset <address> <value>" << std::endl;
    }
}

int get_task(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, Runtime *runtime) {
    int address = *(const int*)task->args;
    return address / 2;
}

void set_task(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, Runtime *runtime) {
    return;
}

int main(int argc, char **argv) {
    Runtime::set_top_level_task_id(DISPATCH_TASK_ID);

    {
        TaskVariantRegistrar registrar(DISPATCH_TASK_ID, "dispatch");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<dispatch_task>(registrar, "dispatch");
    }

    {
        TaskVariantRegistrar registrar(GET_TASK_ID, "get");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<int, get_task>(registrar, "get");
    }

    {
        TaskVariantRegistrar registrar(SET_TASK_ID, "set");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<set_task>(registrar, "set");
    }

    return Runtime::start(argc, argv);
}
