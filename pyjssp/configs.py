# Simulator node types
NOT_START_NODE_SIG = -1
PROCESSING_NODE_SIG = 0
DONE_NODE_SIG = 1
DELAYED_NODE_SIG = 2
DUMMY_NODE_SIG = 3

# Simulator edge types
CONJUNCTIVE_TYPE = 0
DISJUNCTIVE_TYPE = 1

# Simulator edge directions
FORWARD = 0
BACKWARD = 1
NO_DI = -1

# miscellaneous
N_SEP = 1
SEP = ' '
NEW = '\n'

"num_of_machines x num_of_jobs"
benchmarks_name_dict = {"5x10":["la01","la02","la03","la04","la05"],
                        "5x15":["la06","la07","la08","la09","la10"],
                        "5x20":["ft20","la11","la12","la13","la14","la15"],
                        "10x10":["ft10","abz5","abz6","la16","la17","la18","la19","la20","orb01",
                                 "orb02","orb03","orb04","orb05","orb06","orb07","orb08","orb09","orb10"],
                        ####################################################################################
                        "10x15":["la21","la22","la23","la24","la25"],
                        "10x20":["la26","la27","la28","la29","la30","swv01","swv02","swv03","swv04","swv05"],
                        "10x30":["la31","la32","la33","la34","la35"],
                        "10x50":["swv11","swv12","swv13","swv14","swv15","swv16","swv17","swv18","swv19","swv20"],
                        ####################################################################################
                        "15x15":["la36","la37","la38","la39","la40","ta01","ta02","ta03","ta04","ta05",
                                 "ta06","ta07","ta08","ta09","ta10"],
                        "15x20":["abz7","abz8","abz9","swv06","swv07","swv08","swv09","swv10",
                                 "ta11","ta12","ta13","ta14","ta15","ta16","ta17","ta18","ta19","ta20",
                                 "dmu01","dmu02","dmu03","dmu04","dmu05","dmu41","dmu42","dmu43","dmu44","dmu45"],
                        "15x30":["ta31","ta32","ta33","ta34","ta35","ta36","ta37","ta38","ta39","ta40",
                                 "dmu11","dmu12","dmu13","dmu14","dmu15","dmu51","dmu52","dmu53","dmu54","dmu55"],
                        "15x40":["dmu21","dmu22","dmu23","dmu24","dmu25","dmu61","dmu62","dmu63","dmu64","dmu65"],
                        "15x50":["ta51","ta52","ta53","ta54","ta55","ta56","ta57","ta58","ta59","ta60",
                                 "dmu31","dmu32","dmu33","dmu34","dmu35","dmu71","dmu72","dmu73","dmu74","dmu75"],
                        ####################################################################################
                        "20x20":["ta21","ta22","ta23","ta24","ta25","ta26","ta27","ta28","ta29","ta30",
                                 "yn01","yn02","yn03","yn04","dmu06","dmu07","dmu08","dmu09","dmu10",
                                 "dmu46","dmu47","dmu48","dmu49","dmu50"],
                        "20x30":["ta41","ta42","ta43","ta44","ta45","ta46","ta47","ta48","ta49","ta50",
                                 "dmu16","dmu17","dmu18","dmu19","dmu20","dmu56","dmu57","dmu58","dmu59","dmu60"],
                        "20x40":["dmu26","dmu27","dmu28","dmu29","dmu30","dmu66","dmu67","dmu68","dmu69","dmu70"],
                        "20x50":["ta61","ta62","ta63","ta64","ta65","ta66","ta67","ta68","ta69","ta70",
                                 "dmu36","dmu37","dmu38","dmu39","dmu40","dmu76","dmu77","dmu78","dmu79","dmu80"],
                        "20x100":["ta71","ta72","ta73","ta74","ta75","ta76","ta77","ta78","ta79","ta80"]}

if __name__ == "__main__":
    count = 0
    scount = 0
    target_list = []
    for elem_list in benchmarks_name_dict:
            print(elem_list)
            print(benchmarks_name_dict[elem_list])
            for elem in benchmarks_name_dict[elem_list]:
                if "ta" in elem:
                    target_list.append(elem)
                    scount = scount + 1
                count = count + 1
    print(scount)
    print(target_list)
    print(count)

    # process:
    # # 如果是单个目录，就直接写出路径，然后运行
    # # 如果是某个系列instance，返回各个instance 文件名，然后衔接入大路径下，一个一个运行
    # # 如果是 某个instance 大小的，就要用到字典了,