import numpy as np
from pyjssp.configs import (NOT_START_NODE_SIG,
                            PROCESSING_NODE_SIG,
                            DONE_NODE_SIG)


def lastNonZero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    yAxis = np.where(mask.any(axis=axis), val, invalid_val)
    xAxis = np.arange(arr.shape[0], dtype=np.int64)
    xRet = xAxis[yAxis >= 0]
    yRet = yAxis[yAxis >= 0]
    return xRet, yRet


def calEndTimeLB(env,temp1, dur_cp):
    x, y = lastNonZero(temp1, 1, invalid_val=-1)
    dur_cp[np.where(temp1 != 0)] = 0
    #temp2 = dur_cp
    for _,job in env.job_manager.jobs.items():
        job_id = job.job_id
        if job_id in x:
            step_id = y[np.where(x==job.job_id)[0]][0]
            if job[step_id].node_status == DONE_NODE_SIG:
                if step_id+1 < env.num_steps:
                    dur_cp[job_id][step_id+1] = env.global_time + env.processing_time_matrix[job_id][step_id+1]
            elif job[step_id].node_status == PROCESSING_NODE_SIG:
                if temp1[job_id][step_id] > env.global_time:
                    if step_id+1 < env.num_steps:
                        dur_cp[job_id][step_id+1] = temp1[job_id][step_id] + env.processing_time_matrix[job_id][step_id+1]
                elif temp1[job_id][step_id] <= env.global_time:      # occur stochastic processing time disturbance
                    temp1[job_id][step_id] = env.global_time + 2
                    if step_id+1 < env.num_steps:
                        dur_cp[job_id][step_id+1] = temp1[job_id][step_id]+env.processing_time_matrix[job_id][step_id+1]
            else:
                raise "LB update error"
        elif job[0].node_status == NOT_START_NODE_SIG:
            dur_cp[job_id][0] = env.global_time + env.processing_time_matrix[job_id][0]
        else:
            raise "LB update error"
    #dur_cp[x, y] = temp1[x, y]
    temp2 = np.cumsum(dur_cp, axis=1)
    #temp2[np.where(temp1 != 0)] = 0
    ret = temp1+temp2
    return ret


if __name__ == '__main__':
    dur = np.array([[1, 2], [3, 4]])
    temp1 = np.zeros_like(dur)

    temp1[0, 0] = 1
    temp1[1, 0] = 3
    temp1[1, 1] = 5
    print(temp1)

    ret = calEndTimeLB(temp1, dur)
    print(ret)