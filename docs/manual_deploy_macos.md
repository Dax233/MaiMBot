# 📦 macOS系统手动部署MaiMbot麦麦指南

## 准备工作

- 一台搭载了macOS系统的设备（macOS 12.0 或以上）
- QQ小号（QQ框架的使用可能导致qq被风控，严重（小概率）可能会导致账号封禁，强烈不推荐使用大号）
- Homebrew包管理器
    - 如未安装，你可以在https://github.com/Homebrew/brew/releases/latest 找到.pkg格式的安装包
- 可用的大模型API
- 一个AI助手，网上随便搜一家打开来用都行，可以帮你解决一些不懂的问题
- 以下内容假设你对macOS系统有一定的了解，如果觉得难以理解，请直接用Windows系统部署[Windows系统部署指南](./manual_deploy_windows.md)或[使用Windows一键包部署](https://github.com/MaiM-with-u/MaiBot/releases/tag/EasyInstall-windows)
- 终端应用（iTerm2等）

---

## 环境配置

### 1️⃣ **Python环境配置**

```bash
# 检查Python版本（macOS自带python可能为2.7）
python3 --version

# 通过Homebrew安装Python
brew install python@3.12

# 设置环境变量（如使用zsh）
echo 'export PATH="/usr/local/opt/python@3.12/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 验证安装
python3 --version  # 应显示3.12.x
pip3 --version     # 应关联3.12版本
```

### 2️⃣ **创建虚拟环境**

```bash
# 方法1：使用venv（推荐）
python3 -m venv maimbot-venv
source maimbot-venv/bin/activate    # 激活虚拟环境

# 方法2：使用conda
brew install --cask miniconda
conda create -n maimbot python=3.9
conda activate maimbot  # 激活虚拟环境

# 安装项目依赖
# 请确保已经进入虚拟环境再执行
pip install -r requirements.txt
```

---

## 数据库配置

### 3️⃣ **安装MongoDB**

请参考[官方文档](https://www.mongodb.com/zh-cn/docs/manual/tutorial/install-mongodb-on-os-x/#install-mongodb-community-edition)

---

## NapCat

### 4️⃣ **安装与配置Napcat**
- 安装
可以使用Napcat官方提供的[macOS安装工具](https://github.com/NapNeko/NapCat-Mac-Installer/releases/)
由于权限问题，补丁过程需要手动替换 package.json，请注意备份原文件～
- 配置
使用QQ小号登录，添加反向WS地址: `ws://127.0.0.1:8080/onebot/v11/ws`

---

## 配置文件设置

### 5️⃣ **生成配置文件**
可先运行一次
```bash
# 在项目目录下操作
nb run
# 或
python3 bot.py
```

之后你就可以找到`.env.prod`和`bot_config.toml`这两个文件了

关于文件内容的配置请参考：
- [🎀 新手配置指南](./installation_cute.md) - 通俗易懂的配置教程，适合初次使用的猫娘
- [⚙️ 标准配置指南](./installation_standard.md) - 简明专业的配置说明，适合有经验的用户


---

## 启动机器人

### 6️⃣ **启动麦麦机器人**

```bash
# 在项目目录下操作
nb run
# 或
python3 bot.py
```

## 启动管理

### 7️⃣ **通过launchd管理服务**

创建plist文件：

```bash
nano ~/Library/LaunchAgents/com.maimbot.plist
```

内容示例（需替换实际路径）：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.maimbot</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/maimbot-venv/bin/python</string>
        <string>/path/to/MaiMbot/bot.py</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>/path/to/MaiMbot</string>
    
    <key>StandardOutPath</key>
    <string>/tmp/maimbot.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/maimbot.err</string>
    
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

加载服务：

```bash
launchctl load ~/Library/LaunchAgents/com.maimbot.plist
launchctl start com.maimbot
```

查看日志：

```bash
tail -f /tmp/maimbot.log
```

---

## 常见问题处理

1. **权限问题**
```bash
# 遇到文件权限错误时
chmod -R 755 ~/Documents/MaiMbot
```

2. **Python模块缺失**
```bash
# 确保在虚拟环境中
source maimbot-venv/bin/activate  # 或 conda 激活
pip install --force-reinstall -r requirements.txt
```

3. **MongoDB连接失败**
```bash
# 检查服务状态
brew services list
# 重置数据库权限
mongosh --eval "db.adminCommand({setFeatureCompatibilityVersion: '5.0'})"
```

---

## 系统优化建议

1. **关闭App Nap**
```bash
# 防止系统休眠NapCat进程
defaults write NSGlobalDomain NSAppSleepDisabled -bool YES
```

2. **电源管理设置**
```bash
# 防止睡眠影响机器人运行
sudo systemsetup -setcomputersleep Never
```

---
