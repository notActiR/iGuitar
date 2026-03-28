# WebUI 使用说明

## 启动 WebUI

```bash
python scripts/webui.py
```

浏览器会自动打开 http://localhost:7860

## 功能说明

1. **初始化系统** - 启动摄像头和手部追踪
2. **选择歌曲** - 从下拉菜单选择练习曲目
3. **实时反馈** - 绿色表示正确，红色表示错误
4. **练习统计** - 显示正确率和练习次数

## 注意事项

- 首次使用需先运行 `python scripts/calibrate.py` 进行标定
- 确保摄像头权限已开启
- 建议使用 Chrome 或 Edge 浏览器
