#include "stdio.h"
#include "string.h"

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
    char command[4];
    int address, value;
    printf("> ");
    scanf("%3s %d %d", command, &address, &value);
    if (strcmp(command, "get") == 0) {
        printf("Reading address %d...\n", address);
        TaskLauncher launcher(GET_TASK_ID, TaskArgument(&address, sizeof(address)));
        Future get_future = runtime->execute_task(ctx, launcher);
        printf("Value is %d.\n", get_future.get_result<int>());
    } else if (strcmp(command, "set") == 0) {
        printf("Setting address %d to %d...\n", address, value);
    } else {
        printf("Unrecognized command\n");
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
