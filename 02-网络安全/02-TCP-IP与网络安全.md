# TCP/IP 与网络安全

> 理解网络协议栈的安全风险，是一切网络防御的基础。

---

## 四层模型的安全关注点

| 层级 | 代表性协议 | 主要安全风险 | 防御手段 |
|------|-----------|-------------|---------|
| **应用层** | HTTP、DNS、FTP、SMTP | 漏洞利用、协议滥用 | WAF、协议验证、输入清洗 |
| **传输层** | TCP、UDP | SYN Flood、端口扫描、会话劫持 | TCP 代理、SYN Cookie |
| **网络层** | IP、ICMP | IP 欺骗、Smurf 攻击、DDoS | ACL、IP 验证、DDoS 清洗 |
| **链路层** | ARP、MAC | ARP 欺骗、MAC 泛洪 | DHCP Snooping、DAI |

---

## 常见攻击向量

### ARP 欺骗
- 攻击者发送伪造 ARP 报文，冒充网关
- 防御：DHCP Snooping、DAI、静态 ARP

### DNS 劫持
- 伪造 DNS 响应，将用户引导到恶意站点
- 防御：DNSSEC、DoH（DNS over HTTPS）

### TCP SYN Flood
- 发送大量 SYN 包但不完成握手机制，耗尽服务器资源
- 防御：SYN Cookie 注入、连接速率限制

### IP 欺骗
- 伪造源 IP 地址进行攻击
- 防御：入口过滤（Ingress Filtering）、IP 溯源

---

## 协议安全最佳实践

### DNS 安全
- 使用 DoH / DoT 加密 DNS 查询
- 部署 DNSSEC 防止缓存投毒

### HTTP → HTTPS
- HSTS（HTTP Strict Transport Security）
- 强制 TLS 1.2+

### 传输层安全
- 禁用 SSLv3、TLS 1.0/1.1
- 使用强密码套件

---

## 关键协议速查

| 协议 | 端口 | 安全要点 |
|------|------|---------|
| HTTP | 80 | 明文传输，必须升级到 HTTPS |
| HTTPS | 443 | TLS 版本和证书配置是关键 |
| DNS | 53 | 容易被劫持/投毒 |
| SSH | 22 | 禁用密码登录，使用密钥 |
| RDP | 3389 | 不暴露公网，通过 VPN 连接 |

#网络安全 #协议 #TCPIP #DNS #HTTPS #概念
