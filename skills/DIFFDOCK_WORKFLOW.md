# DiffDock 完整工作流程

## 环境信息
- 容器名：oih-diffdock
- 镜像：diffdock:latest
- Python：python3（3.10.12），以 root 运行
- 推理脚本：/app/DiffDock/inference.py
- 工作目录：/app/DiffDock/

## ⚠️ 重要：此镜像非官方rbgcsail/diffdock镜像
- 官方镜像用 micromamba + conda 环境 diffdock（torch 1.13.1+cu117）
- 本镜像用系统 Python（3.10.12）+ pip（torch 2.1.0+cu121）
- 因此不能用 micromamba run -n diffdock，直接用 python3

## ⚠️ 依赖问题（容器重建后需重新安装）
容器重启后以下包会丢失，需重新安装：
```bash
# torch_geometric（已装但重启后需确认）
docker exec oih-diffdock bash -c "pip install torch_geometric 2>&1 | tail -3"

# fair-esm
docker exec oih-diffdock bash -c "pip install fair-esm 2>&1 | tail -3"

# torch geometric 扩展（必须匹配 torch 2.1.0+cu121）
docker exec oih-diffdock bash -c "pip install torch-cluster torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html 2>&1 | tail -5"
```

**永久解决方案**：把以上依赖加入 Dockerfile 重建镜像：
```dockerfile
# 在 /data/docking/Dockerfile.diffdock 中添加
RUN pip install torch_geometric fair-esm && \
    pip install torch-cluster torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

## 模型文件
- 模型位置（容器内）：/app/DiffDock/workdir/v1.1/
- score_model：/app/DiffDock/workdir/v1.1/score_model/best_ema_inference_epoch_model.pt
- confidence_model：/app/DiffDock/workdir/v1.1/confidence_model/best_model_epoch75.pt
- 来源：HuggingFace reginabarzilaygroup/DiffDock-L

### 下载模型命令
```bash
docker exec oih-diffdock bash -c "
cd /app/DiffDock/workdir/v1.1 &&
mkdir -p score_model confidence_model &&
pip install huggingface_hub -q &&
python3 -c \"
from huggingface_hub import hf_hub_download
import shutil
f = hf_hub_download(repo_id='reginabarzilaygroup/DiffDock-L', filename='score_model/best_ema_inference_epoch_model.pt')
shutil.copy(f, 'score_model/')
f = hf_hub_download(repo_id='reginabarzilaygroup/DiffDock-L', filename='confidence_model/best_model_epoch75.pt')
shutil.copy(f, 'confidence_model/')
print('下载完成')
\"
"
```

## ⚠️ GPU注意事项
- NVIDIA_VISIBLE_DEVICES=1 映射宿主机GPU1为容器内GPU0
- DiffDock 自动检测 CUDA，无需手动指定 device
- 显存需求约 8GB

## 推理命令

### 1. 单个蛋白-配体对接
```bash
docker exec oih-diffdock python3 /app/DiffDock/inference.py \
  --protein_path /data/oih/inputs/1IVO.pdb \
  --ligand_description "COc1ccc2c(c1)nc(N)c(C(=O)Nc1ccc(F)cc1)c2" \
  --out_dir /data/oih/outputs/diffdock/ \
  --samples_per_complex 10 \
  --inference_steps 20 \
  --batch_size 4
```

### 2. 批量对接（CSV输入）
```bash
# CSV格式：protein_path,ligand_description,complex_name
cat > /data/oih/inputs/diffdock_input.csv << 'CSV'
protein_path,ligand_description,complex_name
/data/oih/inputs/1IVO.pdb,COc1ccc2c(c1)nc(N)c(C(=O)Nc1ccc(F)cc1)c2,complex_1
CSV

docker exec oih-diffdock python3 /app/DiffDock/inference.py \
  --protein_ligand_csv /data/oih/inputs/diffdock_input.csv \
  --out_dir /data/oih/outputs/diffdock/ \
  --samples_per_complex 10 \
  --inference_steps 20 \
  --batch_size 4
```

### 3. 使用默认配置文件（推荐）
```bash
docker exec oih-diffdock python3 /app/DiffDock/inference.py \
  --config /app/DiffDock/default_inference_args.yaml \
  --protein_path /data/oih/inputs/1IVO.pdb \
  --ligand_description "SMILES_STRING" \
  --out_dir /data/oih/outputs/diffdock/
```

## 关键参数
| 参数 | 说明 | 推荐值 |
|------|------|--------|
| --samples_per_complex | 生成pose数量 | 10~40 |
| --inference_steps | 扩散步数 | 20 |
| --batch_size | 批大小，显存不足时调小 | 4 |
| --no_final_step_noise | 最后一步不加噪声，提高质量 | 加上 |

## 置信度分数解读
- c > 0：高置信度
- -1.5 < c < 0：中等置信度
- c < -1.5：低置信度

## 输出文件结构
```
/data/oih/outputs/diffdock/
└── complex_name/
    ├── rank1_confidence0.xx.sdf    # 最优pose
    ├── rank2_confidence0.xx.sdf
    └── ...
```

## 注意事项
- DiffDock 仅用于小分子-蛋白对接，不适用于蛋白-蛋白
- 配体用 SMILES 字符串输入
- 建议与 GNINA 联用：DiffDock生成pose → GNINA重打分/优化
- 第一次运行会预计算SO(2)/SO(3)分布（约1-2分钟），之后会缓存

## 错误排查
| 错误 | 原因 | 解决 |
|------|------|------|
| ModuleNotFoundError: torch_geometric | 依赖未安装 | 执行上方依赖安装命令 |
| ModuleNotFoundError: esm | fair-esm未安装 | pip install fair-esm |
| ModuleNotFoundError: torch_cluster | torch扩展版本不匹配 | 用cu121版本安装 |
| FileNotFoundError: best_ema_inference_epoch_model.pt | 模型未下载 | 执行模型下载命令 |
| micromamba: command not found | 本镜像非官方版本 | 直接用python3，不用micromamba |

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## ⚠️ 注意事项（自动同步自 CLAUDE.md）

- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
