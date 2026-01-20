import pandas as pd
import os

# 1. 原始 Parquet 文件路径
parquet_path = "/mnt/data/huangbinling/project/occgen/small_vln_n1_split_e2/traj_data/3d/111/trajectory_15/data/chunk-000/episode_000000.parquet"

# 2. 输出 CSV 路径 (保存到当前运行脚本的目录下，方便你找到)
output_csv_path = "./episode_00000_view.csv"

try:
    print(f"⏳ 正在读取 Parquet 文件: {parquet_path} ...")
    
    # 读取数据
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    
    print(f"📊 数据加载成功，包含 {len(df)} 行数据。正在导出 CSV...")
    
    # 导出为 CSV
    # index=False 代表不保存最左边的行号(0, 1, 2...)
    df.to_csv(output_csv_path, index=False)
    
    print(f"✅ 转换成功！")
    print(f"📂 文件位置: {os.path.abspath(output_csv_path)}")
    print("💡 提示：你可以将其下载到本地，用 Excel 打开查看。")

except Exception as e:
    print(f"❌ 转换失败: {e}")
    print("请确保安装了 pandas 和 pyarrow: pip install pandas pyarrow")