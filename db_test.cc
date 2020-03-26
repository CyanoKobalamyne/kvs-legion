#include <cstdint>

#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>

#include "legion.h"

using namespace Legion;

const int MIN_SLEEP_SECONDS = 5;
const int MAX_SLEEP_SECONDS = 10;
const std::string PROMPT = "> ";

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
                   const std::vector<PhysicalRegion> &regions, Context ctx,
                   Runtime *runtime) {
    const InputArgs &command_args = Runtime::get_input_args();
    unsigned int address_count = 0;
    unsigned int read_task_count = 0;
    unsigned int write_task_count = 0;
    for (int i = 1; i < command_args.argc; i++) {
        // Skip any legion runtime configuration parameters
        if (command_args.argv[i][0] == '-') {
            i++;
            continue;
        }

        if (address_count == 0) {
            address_count = atoi(command_args.argv[i]);
        } else if (read_task_count == 0) {
            read_task_count = atoi(command_args.argv[i]);
        } else {
            write_task_count = atoi(command_args.argv[i]);
            break;
        }
    }

    if (address_count == 0 || read_task_count == 0) {
        std::cout << "Usage: " << command_args.argv[0]
                  << " <max_address> <read_tasks> [<write_tasks>]"
                  << std::endl;
    }

    // Define key-value store.
    Rect<1> address_space_bounds(0, address_count);
    IndexSpaceT<1> address_space =
        runtime->create_index_space(ctx, address_space_bounds);
    FieldSpace field_space = runtime->create_field_space(ctx);
    FieldAllocator allocator =
        runtime->create_field_allocator(ctx, field_space);
    allocator.allocate_field(sizeof(value_t), FID_VALUE);
    LogicalRegionT<1> store_region =
        runtime->create_logical_region(ctx, address_space, field_space);

    // Initialize store.
    RegionRequirement init_req(store_region, WRITE_DISCARD, EXCLUSIVE,
                               store_region);
    init_req.add_field(FID_VALUE);
    InlineLauncher init_launcher(init_req);
    PhysicalRegion init_region = runtime->map_region(ctx, init_launcher);
    const FieldAccessor<WRITE_DISCARD, value_t, 1> store(init_region,
                                                         FID_VALUE);
    for (PointInRectIterator<1> iter(address_space_bounds); iter(); iter++) {
        store[*iter] = 0;
    }
    runtime->unmap_region(ctx, init_region);

    // Partition store into individual entries.
    Rect<1> color_bounds = address_space_bounds;
    IndexSpaceT<1> color_space =
        runtime->create_index_space(ctx, color_bounds);
    IndexPartition address_partition =
        runtime->create_equal_partition(ctx, address_space, color_space);
    LogicalPartition store_partition =
        runtime->get_logical_partition(store_region, address_partition);

    // Generate task order.
    std::vector<int> task_nums(read_task_count + write_task_count);
    std::iota(std::begin(task_nums), std::end(task_nums), 0);
    std::random_shuffle(std::begin(task_nums), std::end(task_nums));

    // For generating addresses and values.
    std::default_random_engine rnd_gen;
    std::uniform_int_distribution<address_t> address_dist(
        address_space_bounds.lo, address_space_bounds.hi);
    std::uniform_int_distribution<value_t> value_dist(
        std::numeric_limits<value_t>::min(),
        std::numeric_limits<value_t>::max());

    auto start = std::chrono::high_resolution_clock::now();

    // Start all tasks.
    std::vector<Future> futures;
    for (unsigned int task_num : task_nums) {
        address_t address = address_dist(rnd_gen);
        if (task_num < read_task_count) {
            TaskLauncher launcher(GET_TASK_ID,
                                  TaskArgument(&address, sizeof(address)));
            launcher.add_region_requirement(
                RegionRequirement(runtime->get_logical_subregion_by_color(
                                      store_partition, address),
                                  READ_ONLY, EXCLUSIVE, store_region));
            launcher.add_field(0, FID_VALUE);
            futures.push_back(runtime->execute_task(ctx, launcher));
        } else {
            value_t value = value_dist(rnd_gen);
            Record record = {address = address, value = value};
            TaskLauncher launcher(SET_TASK_ID,
                                  TaskArgument(&record, sizeof(record)));
            launcher.add_region_requirement(
                RegionRequirement(runtime->get_logical_subregion_by_color(
                                      store_partition, address),
                                  WRITE_DISCARD, EXCLUSIVE, store_region));
            launcher.add_field(0, FID_VALUE);
            futures.push_back(runtime->execute_task(ctx, launcher));
        }
    }

    // Wait for all tasks to complete.
    for (Future future : futures) {
        future.get_result<value_t>();
    }

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::cout << duration.count() << std::endl;

    // Free up store.
    runtime->destroy_logical_region(ctx, store_region);
    runtime->destroy_field_space(ctx, field_space);
    runtime->destroy_index_space(ctx, address_space);

    return;
}

value_t get_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime) {
    address_t address = *(address_t *)task->args;
    const FieldAccessor<READ_ONLY, value_t, 1> store(regions[0], FID_VALUE);
    value_t value = store[address];
    return value;
}

value_t set_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime) {
    Record record = *(const Record *)task->args;
    address_t address = record.address;
    value_t value = record.value;
    const FieldAccessor<WRITE_DISCARD, value_t, 1> store(regions[0],
                                                         FID_VALUE);
    store[address] = value;
    return 0;
}

int main(int argc, char **argv) {
    Runtime::set_top_level_task_id(DISPATCH_TASK_ID);

    {
        TaskVariantRegistrar registrar(DISPATCH_TASK_ID, "dispatch");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<dispatch_task>(registrar,
                                                         "dispatch");
    }

    {
        TaskVariantRegistrar registrar(GET_TASK_ID, "get");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<value_t, get_task>(registrar, "get");
    }

    {
        TaskVariantRegistrar registrar(SET_TASK_ID, "set");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<value_t, set_task>(registrar, "set");
    }

    return Runtime::start(argc, argv);
}
