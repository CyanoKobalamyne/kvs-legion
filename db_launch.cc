#include <cstdint>
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

typedef uint8_t address_t;
typedef int64_t value_t;

typedef struct Record {
    address_t address;
    value_t value;
} Record;

void dispatch_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime) {
    std::string line;
    std::string command;
    address_t address;

    std::cout << "> ";
    std::getline(std::cin, line);
    std::stringstream iss(line);
    iss >> command >> address;
    if (command == "get") {
        std::cout << "Reading address" << address << std::endl;
        TaskLauncher launcher(GET_TASK_ID, TaskArgument(&address, sizeof(address)));
        Future future = runtime->execute_task(ctx, launcher);
        std::cout << "Value is: " << future.get_result<value_t>() << std::endl;
    } else if (command == "set") {
        value_t value;
        iss >> value;
        Record record = { address = address, value = value };
        std::cout << "Setting address " << address << " to " << value << std::endl;
        TaskLauncher launcher(SET_TASK_ID, TaskArgument(&record, sizeof(record)));
        Future future = runtime->execute_task(ctx, launcher);
        future.wait();
        std::cout << "Done." << std::endl;
    } else {
        std::cout << "Unrecognized command: " << command << std::endl;
        std::cout << "Allowed commands:" << std::endl;
        std::cout << "\tget <address>" << std::endl;
        std::cout << "\tset <address> <value>" << std::endl;
    }
}

value_t get_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime) {
    address_t address = *(address_t *)task->args;
    return address / 2;
}

void set_task(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, Runtime *runtime) {
    Record record = *(const Record*)task->args;
    address_t address = record.address;
    value_t value = record.value;
    std::cout << "-- " << address << " <= " << value << std::endl;
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
        Runtime::preregister_task_variant<value_t, get_task>(registrar, "get");
    }

    {
        TaskVariantRegistrar registrar(SET_TASK_ID, "set");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<set_task>(registrar, "set");
    }

    return Runtime::start(argc, argv);
}
