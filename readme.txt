商户经营观察服务。现有模块生成特征快照、梯队分类和流动摘要，并把每次处理写入经营账本。

环境：Linux，Python 3.11.2 或更新的兼容版本。所有依赖来自 Python 标准库，无需安装第三方包。

测试：python3 -m unittest discover -s tests/brand_mobility -p 'test_*.py'
构建：python3 -m compileall -q brand_ladder_observatory tests/brand_mobility
验收：python3 -m brand_ladder_observatory.operator.cli fixtures/demo.json --database /tmp/brand_ladder_observatory-acceptance.sqlite3
所有输入均为本地合成数据，运行不需要网络。

目录：brand_ladder_observatory 保存业务处理与存储模块；tests/brand_mobility 保存业务回归；fixtures/demo.json 为原有业务样例，fixtures/analysis 为经营分析资料及字段说明。

原有验收命令每次使用新的 SQLite 路径，避免复用上次状态。新增输入完整性检查：
python3 -m unittest discover -s tests/analysis_inputs -v

经营分群发布（brand_ladder_observatory.segment_release，与既有梯队功能并存）：
  拟合：python3 -m brand_ladder_observatory.segment_release fit --input fixtures/analysis/input.json --database <DB> --output <JSON>
  新客：python3 -m brand_ladder_observatory.segment_release score --input <JSON> --database <DB> --release <发布编号> --output <JSON>
  端到端例子（两次拟合+新客归群，演示重排无迁移、拆分/合并/消失续接、旧版不被覆盖）：python3 examples/segment_release_example.py
  分群测试：python3 -m unittest discover -s tests/segment_release -v

分群方法的业务口径（不是固定分数段改名，而是无监督聚类 + 发布门槛）：
  * 时点与样本：只用 training_cutoff 当天已入库（known_at <= cutoff）的观测；同月多修订取最高 revision；
    每个商户聚合成一份特征，记录多的商户不会主导分群；有效月份不足或核心特征全缺的商户不进训练并在 diagnostics 列名。
  * 特征与尺度：聚类轴为 log1p 月均销售额、月均复购率、月均线上化占比（销售右偏取对数，再按训练集 z-score 标准化，
    参数随发布冻结）；营业稳定性、销售趋势、线上化缺失标记只进画像与解释、不参与距离，避免把单月歇业或数据缺失误切成经营恶化群。
  * 缺失规则：数值按有值月份聚合；线上化占比全缺失时以训练中位数补位并在标准化距离中中立化（置总体典型位置），
    同时保留 digital_missing 标记，"未知"与"纯线下（0）"区分对待。
  * 候选与群数：并行比较 k-means++（球形划分，中心即冻结归群规则）与 Ward 层次凝聚（无随机初始化的独立印证）两种实质不同的方案，
    k 取 2..6，用轮廓系数、最小群规模/占比碎片门槛和跨方案调整兰德指数判断；轮廓不足或只剩碎片群时 status=rejected，明确拒绝发布。
    在最高轮廓系数的简约容忍带内取最小 k，防止缺失值噪声把真实业务群切碎。
  * 发布与沿用：特征变换、缺失规则、群中心、归群规则整体冻结入库（segment_releases 等表），score 只读旧版不重训；
    超支持范围标 out_of_support、缺失严重标 insufficient_data，均不硬塞进群。
  * 续接：重新拟合只通过两个版本共有的商户建立 continuation/split/merge/disappeared/new 关系并保留流向，
    不按编号虚构一对一对应；相同训练内容（含输入重排）按内容指纹幂等复用同一 release_id，不制造迁移；新版插入不覆盖旧版。

容器运行可使用已经预装的 ccusr/prepared:3f007a95fcee-4441968e5ff6 镜像；将当前仓库只读挂载至 /source，在容器内复制到 /tmp/product 后运行上述命令。依赖准备和正式运行分离，运行过程中不需要网络访问。
