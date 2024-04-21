import os

seq_len = 336

print(os.getcwd())
# 创建日志目录
if not os.path.exists("logs"): # 会创建在当前工作目录中
    os.mkdir("logs")
if not os.path.exists("logs/LongForecasting"):
    os.mkdir("logs/LongForecasting")

# 定义运行命令的函数
def run_command(seq_len, pred_len, batch_size, learning_rate):
    command = (
        f"python -u run_longExp.py "
        f"--is_training 0 "
        f"--root_path ./dataset/ "
        f"--data_path exchange_rate.csv "
        f"--model_id Exchange_{seq_len}_{pred_len} "
        f"--model DLinear "
        f"--data custom "
        f"--features M "
        f"--seq_len {seq_len} "
        f"--pred_len {pred_len} "
        f"--enc_in 8 "
        f"--des Exp "
        f"--itr 1 "
        f"--batch_size {batch_size} "
        f"--learning_rate {learning_rate}"
    )
    log_file = f"logs/LongForecasting/DLinear_Exchange_{seq_len}_{pred_len}.log"
    command += f" >{log_file}"
    return command

# 运行命令并输出日志
os.system(run_command(seq_len, 96, 8, 0.0005))
# os.system(run_command(seq_len, 192, 8, 0.0005))
# os.system(run_command(seq_len, 336, 32, 0.0005))
# os.system(run_command(seq_len, 720, 32, 0.005))
