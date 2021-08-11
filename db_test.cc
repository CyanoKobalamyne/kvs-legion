#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>

#include "getopt.h"
#include "legion.h"
#include "x86intrin.h"

#define CPU_FREQ_GHZ 2.6

using namespace Legion;

enum TaskID {
    DISPATCH_TASK_ID,
    GET_TASK_ID,
    SET_TASK_ID,
    TRANSFER_TASK_ID,
};

enum FieldID {
    FID_VALUE,
};

typedef uint16_t address_t;
typedef int64_t value_t;

typedef struct {
    address_t address;
} GetTaskPayload;

typedef struct {
    address_t address;
    value_t value;
} SetTaskPayload;

typedef struct {
    address_t source;
    address_t target;
    value_t amount;
} TransferTaskPayload;

const struct option options[] = {
    {.name = "m", .has_arg = required_argument, .flag = NULL, .val = 'm'},
    {.name = "r", .has_arg = required_argument, .flag = NULL, .val = 'r'},
    {.name = "w", .has_arg = required_argument, .flag = NULL, .val = 'w'},
    {.name = "t", .has_arg = required_argument, .flag = NULL, .val = 't'},
    {0, 0, 0, 0},
};

unsigned long long get_time;
unsigned long long get_count;
unsigned long long set_time;
unsigned long long set_count;
unsigned long long transfer_time;
unsigned long long transfer_count;

void dispatch_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions, Context ctx,
                   Runtime *runtime) {
    const InputArgs &args = Runtime::get_input_args();
    unsigned int address_count = 1;
    unsigned int read_task_count = 0;
    unsigned int write_task_count = 0;
    unsigned int transfer_task_count = 0;

    int opt;
    opterr = 0;
    while ((opt = getopt_long_only(args.argc, args.argv, "", options, NULL)) !=
           -1) {
        switch (opt) {
        case 'm':
            address_count = atoi(optarg);
            break;
        case 'r':
            read_task_count = atoi(optarg);
            break;
        case 'w':
            write_task_count = atoi(optarg);
            break;
        case 't':
            transfer_task_count = atoi(optarg);
            break;
        case '?':
        default:
            break;
        }
    }

    unsigned long total_task_count =
        read_task_count + write_task_count + transfer_task_count;
    if (total_task_count == 0) {
        std::cout << "Usage: " << args.argv[0]
                  << " [-m max_address] [-r read_tasks] [-w write_tasks]"
                     " [-t transfer_tasks]"
                  << std::endl;
        exit(EXIT_FAILURE);
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
    std::vector<int> task_indices(total_task_count);
    std::iota(std::begin(task_indices), std::end(task_indices), 0);
    std::random_shuffle(std::begin(task_indices), std::end(task_indices));

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
    for (unsigned int task_index : task_indices) {
        address_t address = address_dist(rnd_gen);
        if (task_index < read_task_count) {
            GetTaskPayload payload = {.address = address};
            TaskLauncher launcher(
                GET_TASK_ID, TaskArgument(&payload, sizeof(GetTaskPayload)));
            launcher.add_region_requirement(
                RegionRequirement(runtime->get_logical_subregion_by_color(
                                      store_partition, address),
                                  READ_ONLY, EXCLUSIVE, store_region));
            launcher.add_field(0, FID_VALUE);
            futures.push_back(runtime->execute_task(ctx, launcher));
        } else if (task_index < read_task_count + write_task_count) {
            value_t value = value_dist(rnd_gen);
            SetTaskPayload payload = {.address = address, .value = value};
            TaskLauncher launcher(
                SET_TASK_ID, TaskArgument(&payload, sizeof(SetTaskPayload)));
            launcher.add_region_requirement(
                RegionRequirement(runtime->get_logical_subregion_by_color(
                                      store_partition, address),
                                  WRITE_DISCARD, EXCLUSIVE, store_region));
            launcher.add_field(0, FID_VALUE);
            futures.push_back(runtime->execute_task(ctx, launcher));
        } else {
            address_t target = address;
            while (target == address) {
                target = address_dist(rnd_gen);
            }
            value_t amount = value_dist(rnd_gen);
            TransferTaskPayload payload = {
                .source = address, .target = target, .amount = amount};
            TaskLauncher launcher(
                TRANSFER_TASK_ID,
                TaskArgument(&payload, sizeof(TransferTaskPayload)));
            launcher.add_region_requirement(
                RegionRequirement(runtime->get_logical_subregion_by_color(
                                      store_partition, address),
                                  READ_WRITE, EXCLUSIVE, store_region));
            launcher.add_field(0, FID_VALUE);
            launcher.add_region_requirement(
                RegionRequirement(runtime->get_logical_subregion_by_color(
                                      store_partition, target),
                                  READ_WRITE, EXCLUSIVE, store_region));
            launcher.add_field(1, FID_VALUE);
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
    std::cout << "Time: " << duration.count() << " ns" << std::endl;

    std::cout << std::fixed << std::setprecision(0);
    std::cout << "GET tasks took " << (get_time / CPU_FREQ_GHZ / get_count)
              << " ns on average\n";
    std::cout << "SET tasks took " << (set_time / CPU_FREQ_GHZ / set_count)
              << " ns on average\n";
    std::cout << "TRANSFER tasks took "
              << (transfer_time / CPU_FREQ_GHZ / transfer_count)
              << " ns on average\n";

    // Free up store.
    runtime->destroy_logical_region(ctx, store_region);
    runtime->destroy_field_space(ctx, field_space);
    runtime->destroy_index_space(ctx, address_space);

    return;
}

value_t get_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime) {
    unsigned long long start = __rdtsc();
    GetTaskPayload payload = *(const GetTaskPayload *)task->args;
    address_t address = payload.address;
    const FieldAccessor<READ_ONLY, value_t, 1> store(regions[0], FID_VALUE);
    value_t value = store[address];
    unsigned long long end = __rdtsc();
    std::cerr << "[GET] took " << (end - start) / CPU_FREQ_GHZ << ", value "
              << value << std::endl;
    get_count++;
    get_time += end - start;
    return 0;
}

value_t set_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime) {
    unsigned long long start = __rdtsc();
    SetTaskPayload payload = *(const SetTaskPayload *)task->args;
    address_t address = payload.address;
    value_t value = payload.value;
    const FieldAccessor<WRITE_DISCARD, value_t, 1> store(regions[0],
                                                         FID_VALUE);
    store[address] = value;
    unsigned long long end = __rdtsc();
    std::cerr << "[SET] took " << (end - start) / CPU_FREQ_GHZ << std::endl;
    set_count++;
    set_time += end - start;
    return 0;
}

value_t transfer_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions, Context ctx,
                      Runtime *runtime) {
    unsigned long long start = __rdtsc();
    TransferTaskPayload payload = *(const TransferTaskPayload *)task->args;
    address_t source = payload.source;
    address_t target = payload.target;
    value_t amount = payload.amount;
    const FieldAccessor<READ_WRITE, value_t, 1> source_store(regions[0],
                                                             FID_VALUE);
    const FieldAccessor<READ_WRITE, value_t, 1> target_store(regions[1],
                                                             FID_VALUE);
    value_t source_val = source_store[source];
    value_t target_val = target_store[target];
    if (amount <= source_val) {
        source_store[source] = source_val - amount;
        target_store[target] = target_val + amount;
    } else {
        source_store[source] = 0;
        target_store[target] = target_val + source_val;
    }
    unsigned long long end = __rdtsc();
    std::cerr << "[TRANSFER] took " << (end - start) / CPU_FREQ_GHZ
              << std::endl;
    transfer_count++;
    transfer_time += end - start;
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

    {
        TaskVariantRegistrar registrar(TRANSFER_TASK_ID, "transfer");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<value_t, transfer_task>(registrar,
                                                                  "transfer");
    }

    return Runtime::start(argc, argv);
}
