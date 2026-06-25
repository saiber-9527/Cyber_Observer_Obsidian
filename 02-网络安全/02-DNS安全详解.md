# DNS 安全详解

> DNS 是互联网的"电话本"，也是攻击者的重点目标。

---

## DNS 攻击面

```
用户设备
  │
  ├── DNS 劫持（中间人篡改 DNS 响应）
  ├── DNS 投毒（缓存污染）
  ├── DNS 隧道（利用 DNS 协议做 C2 通信）
  ├── DNS 放大攻击（反射放大 DDoS）
  ├── DNS 重绑定（绕过同源策略）
  └── 域名抢注/仿冒（Typosquatting）
```

---

## 常见攻击详解

### 1. DNS 劫持
攻击者篡改 DNS 解析结果，将用户引导到恶意站点。

```
正常：用户 → baidu.com → 正确的 IP → 真正的百度
劫持：用户 → baidu.com → 攻击者的 IP → 钓鱼网站
```

**防御：**
- 使用 DNSSEC 验证 DNS 响应的真实性
- 使用 DoH（DNS over HTTPS）/ DoT（DNS over TLS）
- 配置可信的 DNS 服务器

### 2. DNS 投毒（缓存污染）
向 DNS 服务器注入错误的 DNS 记录，影响所有使用该 DNS 的用户。

```
攻击者向 DNS 服务器发送伪造的响应
DNS 服务器缓存了错误的记录
所有用户解析该域名时都被指向恶意站点
```

**防御：**
- 使用 DNSSEC（签名 DNS 记录）
- 限制 DNS 递归解析的来源
- 配置 Source Port Randomization

### 3. DNS 隧道
攻击者将非 DNS 流量（如 C2 通信）封装在 DNS 查询和响应中。

```
被控主机 → DNS 查询 "abc123.evil.com" → 攻击者 DNS 服务器
         ← DNS 响应中携带指令/工具
```

**特点：**
- 防火墙一般不会封 DNS 53 端口
- 带宽小但隐蔽性强
- 常用于突破防火墙的出口限制

**检测：**
- DNS 查询频率异常
- DNS 查询长度异常（正常 DNS 查询很短）
- TXT 记录响应体积异常大

### 4. DNS 放大攻击
攻击者用伪造的源 IP（受害者 IP）向开放 DNS 服务器发送小查询，DNS 服务器向受害者返回大量数据。

```
攻击者（伪造源 IP = 受害者）
    │
    小查询 (60 bytes) → 开放 DNS 解析器
                          │
                          大响应 (4000 bytes) → 受害者
                          放大倍数 ≈ 60~70x
```

---

## DNS 安全协议

| 协议 | 说明 | 端口 |
|------|------|------|
| **DNS over UDP** | 传统 DNS，53 端口 | 明文 |
| **DNS over TCP** | 大数据传输，53 端口 | 明文 |
| **DNSSEC** | DNS 数据签名认证 | 53（+签名数据） |
| **DNS over TLS（DoT）** | TLS 加密传输 | 853 |
| **DNS over HTTPS（DoH）** | HTTPS 加密传输 | 443 |
| **DNS over QUIC** | 基于 QUIC 加密传输 | 443 |

---

## DNS 配置最佳实践

### 企业 DNS 服务器配置
```bash
# BIND 安全配置要点
options {
    # 限制递归查询来源
    allow-query { trusted; };
    allow-recursion { trusted; };
    
    # 限制 zone 传输
    allow-transfer { slave-servers; };
    
    # 防止 DNS 放大攻击
    allow-query-cache { trusted; };
    
    # 关闭递归查询（对外）
    recursion yes;
    allow-recursion { internal-net; };
    
    # DNS 安全扩展
    dnssec-validation auto;
};

# 日志
logging {
    channel security_log {
        file "/var/log/named/security.log" versions 3 size 10m;
        severity info;
        print-time yes;
    };
    category security { security_log; };
};
```

### 客户端配置
```bash
# 使用 DoH/DoT（以 Cloudflare 为例）
# /etc/resolv.conf
nameserver 1.1.1.1
nameserver 1.0.0.1
options edns0

# 安装 stubby（DoT 客户端）或使用 firefox/chrome 内置 DoH
```

---

## 常用命令

```bash
# DNS 查询
dig example.com ANY
nslookup example.com
host example.com

# 反向查询
dig -x 8.8.8.8

# 查看 DNS 解析路径
dig +trace example.com

# 查询特定类型
dig example.com MX
dig example.com TXT

# DNSSEC 验证
dig +dnssec example.com
```

---

## ⚠️ 常见问题

1. **DNS 配置错误可能导致业务中断** — 修改 DNS 记录要谨慎
2. **DNSSEC 配置复杂** — 密钥管理和签名流程需要学习成本
3. **DoH/DoT 可能绕过企业安全策略** — 用户配置了 DoH 后，企业 DNS 过滤器就失效了
4. **DNS 日志占用空间大** — 需要规划存储和提取策略

#网络安全 #DNS #协议 #最佳实践
