# Quantify A-Stock

第一版是 A 股研究型选股系统，不做自动交易。系统用长期 K 线序列、成交量结构、个股财报特征、板块环境和大盘环境，预测下一交易日涨幅是否超过配置阈值，并输出每日 Top 20 候选股。

## 设计边界

- 股票池：A 股全市场。
- 目标：下一交易日收盘涨幅是否超过阈值，默认 `1%`。
- 模型：PyTorch GRU 深度序列模型，另带轻量基线模型用于对照。
- 防泄露：训练/验证/测试按时间切分；财报特征按生效日期向后匹配；特征列排除 `future_close`、`next_return`、`label`。
- GitHub：只同步代码、配置样例和文档；`data/`、`models/`、`reports/` 不提交。

## 本地快速验证

```powershell
pip install -r requirements.txt
.\scripts\smoke_sample.ps1
```

这会生成样例数据，训练一个小模型，并在 `reports/` 下输出 Top N CSV。

## 远程电脑运行

远程电脑 `100.114.117.105` 通过远程桌面使用。建议目录：

```text
D:\Quantify\repo
D:\Quantify\data
D:\Quantify\models
D:\Quantify\reports
```

首次在远程电脑：

```powershell
cd D:\Quantify
git clone <你的私有GitHub仓库URL> repo
cd repo
pip install -r requirements.txt
Copy-Item configs\config.remote.example.yaml configs\config.local.yaml
```

之后每次运行：

```powershell
cd D:\Quantify\repo
.\scripts\run_remote_daily.ps1
```

## 数据文件约定

- `data/stocks/daily.csv`：个股日线，至少包含 `date, code, open, high, low, close, volume, amount`。
- `data/market/index_daily.csv`：指数日线，用于大盘状态。
- `data/market/sector_daily.csv`：行业/板块日线，用于板块强弱。
- `data/fundamentals/features.csv`：长表财报特征，字段为 `date, code, feature, value, source_sheet`。

My_Stock_v1.0 的 `std.xlsx` 和最终工作簿用于提取财报特征。若没有真实披露日期，当前实现使用保守滞后天数，默认 90 天，避免财报按报表期提前进入模型。

## 常用命令

```powershell
python -m quantify.cli --config configs\config.local.yaml fetch-stock-list
python -m quantify.cli --config configs\config.local.yaml fetch-daily --limit 20
python -m quantify.cli --config configs\config.local.yaml fetch-index
python -m quantify.cli --config configs\config.local.yaml extract-my-stock --list-std
python -m quantify.cli --config configs\config.local.yaml extract-my-stock
python -m quantify.cli --config configs\config.local.yaml train
python -m quantify.cli --config configs\config.local.yaml predict
```

## 本机每日更新

真实数据首次初始化时只选择深沪主板 `00/60`，股票代码始终保存为六位字符串：

```powershell
$env:PYTHONPATH="src"
python -m quantify.cli --config configs\config.local.yaml fetch-stock-list
python -m quantify.cli --config configs\config.local.yaml fetch-index --workers 4
python -m quantify.cli --config configs\config.local.yaml fetch-daily --prefixes "00,60" --workers 4
python -m quantify.cli --config configs\config.local.yaml train
python -m quantify.cli --config configs\config.local.yaml predict
```

初始历史数据建立后，每天执行：

```powershell
.\scripts\run_local_daily.ps1
```

当前脚本默认只维护 `00/60` 各 40 只的真实验证池，避免在第一版模型尚未改成流式训练前误把全池样本一次性装入内存。

需要重新训练模型时：

```powershell
.\scripts\run_local_daily.ps1 -Retrain
```

每日增量流程会回补最近 7 个自然日并按 `code,date` 合并去重，同时加入当日股票池的上涨比例、收益中位数、横截面波动和总成交额变化等市场状态特征。
