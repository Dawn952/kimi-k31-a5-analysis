# A5 公开分析产物维护

- 默认中文。仅维护本轮 A5 权重、通信、资源泳道与 DCP 对照材料。
- 用户要求：本地修改完成后，同步到本仓库 GitHub 远端，并核对 GitHub Pages 最新构建与实际页面；不要只交付本地副本。
- 发布验收同时核对远端最新commit、Actions与github-pages deployment的SHA/状态、匿名HTTP访问；不能只看Pages配置已创建。
- `src/` 是此公开版本的源码；修改后运行 `node scripts/build.mjs` 重建 `docs/index.html`，对应静态图也须更新。
- `node --test tests/*.cjs` 验证公式、依赖、资源框线和公开的数据片段。
- 本仓库公开。发布内容限定为分析代码、图示、必要的精简统计；不要加入凭据、机器地址、账号目录、原始 trace、权重文件或运行环境的私有源码。
- 明确区分源码事实、DCP 设备区间观察、新拓扑理论估算和待实现方案。通信区间交集不等于端到端收益。
- KDA 使用资源泳道，同框展示 MX quant/beta BF16、QKV返回/G GEMM、G返回/Recurrent 等跨引擎窗口；不把两个 Vector/Cube 算子无依赖就称为有效掩盖。
