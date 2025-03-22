# 📦 Linux系统如何手动部署MaiMbot麦麦？

## 准备工作

- 一台联网的Linux设备（本教程以Ubuntu/Debian系为例）
- QQ小号（QQ框架的使用可能导致qq被风控，严重（小概率）可能会导致账号封禁，强烈不推荐使用大号）
- 可用的大模型API
- 一个AI助手，网上随便搜一家打开来用都行，可以帮你解决一些不懂的问题
- 以下内容假设你对Linux系统有一定的了解，如果觉得难以理解，请直接用Windows系统部署[Windows系统部署指南](./manual_deploy_windows.md)或[使用Windows一键包部署](https://github.com/MaiM-with-u/MaiBot/releases/tag/EasyInstall-windows)

## 你需要知道什么？

- 如何正确向AI助手提问，来学习新知识

- Python是什么

- Python的虚拟环境是什么？如何创建虚拟环境

- 命令行是什么

- 数据库是什么？如何安装并启动MongoDB

- 如何运行一个QQ机器人，以及NapCat框架是什么

---

## 一键部署
请下载并运行项目根目录中的run.sh并按照提示安装，部署完成后请参照后续配置指南进行配置

## 环境配置

### 1️⃣ **确认Python版本**

需确保Python版本为3.9及以上

```bash
python --version
# 或
python3 --version
```

如果版本低于3.9，请更新Python版本，目前建议使用python3.12

```bash
# Debian
sudo apt update
sudo apt install python3.12
# Ubuntu
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12

# 执行完以上命令后，建议在执行时将python3指向python3.12
# 更新替代方案，设置 python3.12 为默认的 python3 版本:
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo update-alternatives --config python3
```
建议再执行以下命令，使后续运行命令中的`python3`等同于`python`
```bash
sudo apt install python-is-python3
```

### 2️⃣ **创建虚拟环境**

```bash
# 方法1：使用venv(推荐)
python3 -m venv maimbot
source maimbot/bin/activate  # 激活环境

# 方法2：使用conda（需先安装Miniconda）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n maimbot python=3.9
conda activate maimbot

# 通过以上方法创建并进入虚拟环境后，再执行以下命令

# 安装依赖（任选一种环境）
pip install -r requirements.txt
```

---

## 数据库配置

### 3️⃣ **安装并启动MongoDB**

- 安装与启动：请参考[官方文档](https://www.mongodb.com/zh-cn/docs/manual/administration/install-on-linux/#std-label-install-mdb-community-edition-linux)，进入后选择自己的系统版本即可
- 默认连接本地27017端口

---

## NapCat配置

### 4️⃣ **安装NapCat框架**

- 执行NapCat的Linux一键使用脚本(支持Ubuntu 20+/Debian 10+/Centos9) 
```bash
curl -o napcat.sh https://nclatest.znin.net/NapNeko/NapCat-Installer/main/script/install.sh && sudo bash napcat.sh
```
- 如果你不想使用Napcat的脚本安装，可参考[Napcat-Linux手动安装](https://www.napcat.wiki/guide/boot/Shell-Linux-SemiAuto)

-  使用QQ小号登录，添加反向WS地址: `ws://127.0.0.1:8080/onebot/v11/ws`

---

## 配置文件设置

### 5️⃣ **配置文件设置，让麦麦Bot正常工作**
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

---

### 7️⃣ **使用systemctl管理maimbot**

使用以下命令添加服务文件：

```bash
sudo nano /etc/systemd/system/maimbot.service
```

输入以下内容：

`<maimbot_directory>`：你的maimbot目录

`<venv_directory>`：你的venv环境（就是上文创建环境后，执行的代码`source maimbot/bin/activate`中source后面的路径的绝对路径）

```ini
[Unit]
Description=MaiMbot 麦麦
After=network.target mongod.service

[Service]
Type=simple
WorkingDirectory=<maimbot_directory>
ExecStart=<venv_directory>/python3 bot.py
ExecStop=/bin/kill -2 $MAINPID
Restart=always
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

输入以下命令重新加载systemd：

```bash
sudo systemctl daemon-reload
```

启动并设置开机自启：

```bash
sudo systemctl start maimbot
sudo systemctl enable maimbot
```

输入以下命令查看日志：

```bash
sudo journalctl -xeu maimbot
```

---

## **其他组件(可选)**

- 直接运行 knowledge.py生成知识库

---

## 常见问题

🔧 权限问题：在命令前加`sudo`  
🔌 端口占用：使用`sudo lsof -i :8080`查看端口占用  
🛡️ 防火墙：确保8080/27017端口开放  

```bash
sudo ufw allow 8080/tcp
sudo ufw allow 27017/tcp
```
