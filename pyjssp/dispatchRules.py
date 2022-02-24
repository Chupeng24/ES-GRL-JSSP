import numpy as np

def dispatch_rule_func(env,machine,disrule_name):
    action = None
    if disrule_name == "SPT":
        # op_id = random.choice(machine.doable_ops_id)
        shortest_process_time = float("inf")
        for op_id in machine.doable_ops_id:
            job_id, step_id = env.job_manager.sur_index_dict[op_id]
            operation = env.job_manager[job_id][step_id]
            if shortest_process_time > operation.processing_time:
                action = operation
                shortest_process_time = operation.processing_time
    elif disrule_name == "LPT":
        longest_process_time = 0
        for op_id in machine.doable_ops_id:
            job_id, step_id = env.job_manager.sur_index_dict[op_id]
            operation = env.job_manager[job_id][step_id]
            if longest_process_time < operation.processing_time:
                action = operation
                longest_process_time = operation.processing_time
    elif disrule_name == "FIFO":
        min_arrive_time = float("inf")
        for op_id in machine.doable_ops_id:
            job_id, step_id = env.job_manager.sur_index_dict[op_id]
            operation = env.job_manager[job_id][step_id]
            if min_arrive_time > operation.arrive_time:
                action = operation
                min_arrive_time = operation.arrive_time
    elif disrule_name == "LIFO":
        max_arrive_time = -1
        for op_id in machine.doable_ops_id:
            job_id, step_id = env.job_manager.sur_index_dict[op_id]
            operation = env.job_manager[job_id][step_id]
            if max_arrive_time < operation.arrive_time:
                action = operation
                max_arrive_time = operation.arrive_time
    elif disrule_name == "JSPT" or disrule_name == "STPT":
        shortest_process_time = float("inf")
        for op_id in machine.doable_ops_id:
            job_id, step_id = env.job_manager.sur_index_dict[op_id]
            job = env.job_manager[job_id]
            if shortest_process_time > job.processing_time:
                operation = env.job_manager[job_id][step_id]
                action = operation
                shortest_process_time = job.processing_time
    elif disrule_name == "JLPT" or disrule_name == "LTPT":
        longest_process_time = 0
        for op_id in machine.doable_ops_id:
            job_id, step_id = env.job_manager.sur_index_dict[op_id]
            job = env.job_manager[job_id]
            if longest_process_time < job.processing_time:
                operation = env.job_manager[job_id][step_id]
                action = operation
                longest_process_time = job.processing_time
    elif disrule_name == "LOR":
        least_rem_op_num = float("inf")
        for op_id in machine.doable_ops_id:
            job_id, step_id = env.job_manager.sur_index_dict[op_id]
            operation = env.job_manager[job_id][step_id]
            if least_rem_op_num > operation.remaining_ops:
                action = operation
                least_rem_op_num = operation.remaining_ops
    elif disrule_name == "MOR":
        most_rem_op_num = -1
        for op_id in machine.doable_ops_id:
            job_id, step_id = env.job_manager.sur_index_dict[op_id]
            operation = env.job_manager[job_id][step_id]
            if most_rem_op_num < operation.remaining_ops:
                action = operation
                most_rem_op_num = operation.remaining_ops
    elif disrule_name == "MWKR":
        most_work_rem = -1
        for op_id in machine.doable_ops_id:
            job_id, step_id = env.job_manager.sur_index_dict[op_id]
            operation = env.job_manager[job_id][step_id]
            op_remain_work = operation.processing_time
            temp_op = operation
            while temp_op.next_op_built:
                temp_op = temp_op.next_op
                op_remain_work = op_remain_work + temp_op.processing_time
            if most_work_rem < op_remain_work:
                action = operation
                most_work_rem = operation.remaining_ops
    elif disrule_name == "FDD/MWKR":
        min_ratio = float("inf")
        for op_id in machine.doable_ops_id:
            job_id, step_id = env.job_manager.sur_index_dict[op_id]
            operation = env.job_manager[job_id][step_id]
            job = env.job_manager[job_id]
            Re = 0
            # obtain operation remaining work
            op_remain_work = operation.processing_time
            temp_op = operation
            while temp_op.next_op_built:
                temp_op = temp_op.next_op
                op_remain_work = op_remain_work + temp_op.processing_time
            # obtain ratio of Flow Due Date to Most Work Remaining
            FDD = (Re + (job.processing_time - op_remain_work) + operation.processing_time)
            ratio_FDD_to_MWKR = FDD/op_remain_work

            if min_ratio > ratio_FDD_to_MWKR:
                action = operation
                min_ratio = ratio_FDD_to_MWKR
    return action