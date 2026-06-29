# 获取当前脚本的目录
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 获取传递给脚本的文件名参数
file_name="env.yaml"

# 构建文件的相对路径
file_path="$script_dir/$file_name"

det shell start --config-file $file_path