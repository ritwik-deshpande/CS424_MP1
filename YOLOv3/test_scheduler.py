import time
from scheduling.Scheduler import *


start = time.time()

scheduler = Scheduler()
scheduler.run()
scheduler.print_history()
scheduler.visualize_history()

end = time.time()

print("Elapsed time: %f s" % (end - start))


# # example for using visualize_history_file()
# history = read_json_file("scheduling_history.json")
# visualize_history_file(history)
# # calculate group average response time from history file
# group_response_time = get_group_avg_response_time(history)
# print(group_response_time)
