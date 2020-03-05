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

enum FieldID {
    FID_VALUE,
};

typedef uint16_t address_t;
typedef int64_t value_t;

typedef struct Record {
    address_t address;
    value_t value;
} Record;

void dispatch_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime) {
    // Define key-value store.
    Rect<1> address_space_bounds(std::numeric_limits<address_t>::min(),
                                 std::numeric_limits<address_t>::max());
    IndexSpaceT<1> address_space = runtime->create_index_space(ctx, address_space_bounds);
    FieldSpace field_space = runtime->create_field_space(ctx);
    FieldAllocator allocator = runtime->create_field_allocator(ctx, field_space);
    allocator.allocate_field(sizeof(value_t), FID_VALUE);
    LogicalRegionT<1> store_region = runtime->create_logical_region(ctx, address_space, field_space);

    // Initialize store.
    RegionRequirement init_req(store_region, READ_WRITE, EXCLUSIVE, store_region);
    init_req.add_field(FID_VALUE);
    InlineLauncher init_launcher(init_req);
    PhysicalRegion init_region = runtime->map_region(ctx, init_launcher);
    const FieldAccessor<READ_WRITE, value_t, 1> store(init_region, FID_VALUE);
    for (PointInRectIterator<1> iter(address_space_bounds); iter(); iter++) {
        store[*iter] = 0;
    }
    runtime->unmap_region(ctx, init_region);


    // Display prompt.
    std::cout << "> ";

    // Read line.
    std::string line;
    std::getline(std::cin, line);

    // Parse command.
    std::stringstream iss(line);
    std::string command;
    address_t address;
    iss >> command >> address;
    if (command == "get") {
        std::cout << "Reading address " << address << std::endl;
        TaskLauncher launcher(GET_TASK_ID, TaskArgument(&address, sizeof(address)));
        launcher.add_region_requirement(RegionRequirement(store_region, READ_ONLY, EXCLUSIVE, store_region));
        launcher.add_field(0, FID_VALUE);
        Future future = runtime->execute_task(ctx, launcher);
        std::cout << "Value is: " << future.get_result<value_t>() << std::endl;
    } else if (command == "set") {
        value_t value;
        iss >> value;
        Record record = { address = address, value = value };
        std::cout << "Setting address " << address << " to " << value << std::endl;
        TaskLauncher launcher(SET_TASK_ID, TaskArgument(&record, sizeof(record)));
        launcher.add_region_requirement(RegionRequirement(store_region, WRITE_DISCARD, EXCLUSIVE, store_region));
        launcher.add_field(0, FID_VALUE);
        Future future = runtime->execute_task(ctx, launcher);
        future.wait();
        std::cout << "Done." << std::endl;
    } else {
        std::cout << "Unrecognized command: " << command << std::endl;
        std::cout << "Allowed commands:" << std::endl;
        std::cout << "\tget <address>" << std::endl;
        std::cout << "\tset <address> <value>" << std::endl;
    }

    // Free up store.
    runtime->destroy_logical_region(ctx, store_region);
    runtime->destroy_field_space(ctx, field_space);
    runtime->destroy_index_space(ctx, address_space);
}

value_t get_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime) {
    address_t address = *(address_t *)task->args;
    const FieldAccessor<READ_ONLY, value_t, 1> store(regions[0], FID_VALUE);
    value_t value = store[address];
    return value;
}

void set_task(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, Runtime *runtime) {
    Record record = *(const Record*)task->args;
    address_t address = record.address;
    value_t value = record.value;
    const FieldAccessor<WRITE_DISCARD, value_t, 1> store(regions[0], FID_VALUE);
    store[address] = value;
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
