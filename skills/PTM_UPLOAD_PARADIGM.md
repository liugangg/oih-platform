# PTM Upload Paradigm — Qwen 决策范式

## 当收到带 [用户上传文件] 标记的 message 时

### 规则1：优先使用已生成的输入文件，不要自己重建
- 看到 af3_input.json 路径 → 直接传给 alphafold3 工具，不要重新写 JSON
- 看到 gromacs_ptm_notes.json → 读取里面的 force_field 和 disulfide_pairs 参数
- 看到 adc_input.json → 传给 freesasa，conjugation_sites 已预设

### 规则2：PTM 与工具的对应关系
| PTM类型 | 可用工具 | 不可用工具 | 注意事项 |
|--------|---------|---------|---------|
| 糖基化(NAG/FUC) | AlphaFold3 | GROMACS | GROMACS缺GLYCAM力场，只做AF3预测 |
| 磷酸化(SEP/TPO) | AlphaFold3 | GROMACS(限制) | GROMACS需CHARMM36，告知用户精度有限 |
| 二硫键(SSBOND) | AF3+GROMACS+RFdiffusion | — | GROMACS加-ss参数，RFdiffusion加--disulfide |
| Cys偶联位点 | freesasa+rdkit_conjugate | — | 先freesasa确认可及性再rdkit_conjugate |
| 不支持的PTM | — | 所有工具 | 明确告知用户该PTM平台暂不支持 |

### 规则3：FASTA 序列上传
- 没有结构 → 先调 alphafold3 预测结构，再进行后续分析
- 有结构需要设计突变 → 调 proteinmpnn 或 rfdiffusion

### 规则4：SMILES 上传
- 小分子配体 → 先调 chemprop 预测ADMET，再调 gnina/autodock 对接
- ADC payload → 调 rdkit_conjugate，smiles 直接传入

### 规则5：不确定时
- PTM 类型不认识 → 报告给用户，说明平台不支持，建议移除后重新上传
- 文件解析失败 → 报告具体错误，不要猜测

### 典型 workflow 示例

**用户上传含糖基化抗体PDB + "预测与CD36的结合"**
1. 读取 af3_input.json（已包含糖基化配体）
2. 调用 alphafold3(input_json=af3_input.json路径)
3. 用 AF3 输出结构调用 fpocket 找结合口袋
4. 调用 gnina 对接
5. 汇报结合能 + ipTM

**用户粘贴 FASTA + "设计纳米抗体"**
1. 调用 alphafold3 预测靶点结构（FASTA → 结构）
2. 调用 rfdiffusion(target_pdb=预测结构, mode=binder_design)
3. 调用 proteinmpnn 优化序列
4. 调用 alphafold3 验证复合物 ipTM
