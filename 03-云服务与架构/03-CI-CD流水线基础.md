# CI/CD 流水线基础

> 持续集成/持续部署——现代云原生开发的标配。

---

## 什么是 CI/CD

```
CI（持续集成）：
  开发 → 提交代码 → 自动构建 → 自动测试 → 发现问题立即反馈
  （频繁的集成和验证，防止"在我电脑上是好的"）

CD（持续部署/交付）：
  通过测试的代码 → 自动部署到预发布环境 → 自动部署到生产环境
  （快速、可靠、可重复的发布流程）
```

---

## 典型流水线

```yaml
# 一个完整的 CI/CD 流水线示例（GitHub Actions）

name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # 1. 代码检查和安全扫描
  lint-and-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: SAST（静态代码分析）
        run: |
          pip install bandit
          bandit -r src/ -f json -o sast-report.json
      - name: SCA（依赖扫描）
        run: |
          pip install safety
          safety check -r requirements.txt
      - name: Secret 扫描
        uses: trufflesecurity/trufflehog@v3
        with:
          extra_args: --results=verified,unknown

  # 2. 构建和单元测试
  build-and-test:
    needs: lint-and-scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker Image
        run: docker build -t myapp:${{ github.sha }} .
      - name: Run Unit Tests
        run: docker run myapp:${{ github.sha }} pytest
      - name: Scan Image for CVEs
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          format: table
          exit-code: 1

  # 3. 部署
  deploy:
    needs: build-and-test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Production
        run: |
          # 推送镜像到仓库
          docker push registry.example.com/myapp:${{ github.sha }}
          # 更新 K8s 部署
          kubectl set image deployment/myapp \
            myapp=registry.example.com/myapp:${{ github.sha }}
```

---

## 流水线各阶段的安全控制

| 阶段 | 安全措施 | 工具 |
|------|---------|------|
| **代码提交** | 代码审查（PR）+ 禁止直接 push main | GitHub/GitLab |
| **代码扫描** | SAST（静态安全测试） | SonarQube、Semgrep、CodeQL |
| **依赖扫描** | SCA（软件成分分析） | Snyk、Trivy、OWASP Dependency-Check |
| **构建** | 使用安全的基础镜像 | Docker Bench、Trivy |
| **镜像扫描** | 容器镜像 CVE 扫描 | Trivy、Clair、Anchore |
| **部署前** | IaC 安全扫描 | Checkov、tfsec |
| **部署** | 蓝绿部署/金丝雀发布 | K8s Rollout、Spinnaker |
| **部署后** | 运行时监控 | Falco、Datadog |

---

## 安全左移（Shift Left）

```
传统做法：
  开发 → 测试 → 安全扫描 → 部署
                     ↑
              安全在这里发现问题→开发重做

Shift Left 做法：
  开发 → 安全扫描 → 测试 → 部署
           ↑
     开发中就能发现安全问题→立即修复

目标：在越早的阶段发现安全问题，修复成本越低。
```

---

## 常见 CI/CD 安全漏洞

| 漏洞 | 说明 | 防御 |
|------|------|------|
| Secret 泄露 | 代码仓库中硬编码了 API Key | 使用 Key Vault / GitHub Secrets |
| 恶意依赖 | 使用被投毒的第三方包 | SCA 扫描 + 版本锁定 |
| 镜像投毒 | 基础镜像中含有恶意软件 | 使用可信镜像 + 签名验证 |
| 权限过大 | CI Token 权限太大 | 最小权限的 CI Token |
| 不安全的 CD | 自动部署到生产环境无人工审批 | 生产部署增加人工审批节点 |

---

## ⚠️ 常见陷阱

1. **CI 通过了但本地跑不了** — 环境不一致，用容器化解决
2. **测试覆盖不全** — CI 绿灯大但没几行测试代码
3. **安全扫描只在 CI 中做一次** — 依赖更新是持续过程，需要定期扫描
4. **CD 太"激进"** — 没有灰度发布直接全量替换，出问题就全挂

#云计算 #CI/CD #DevOps #概念
