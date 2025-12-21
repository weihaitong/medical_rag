#!/usr/bin/env python3
# check_neo4j.py - 检测 Neo4j 连通性（兼容 neo4j 5.x+）
import os
import sys
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable  # ← 关键修改

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://192.168.43.225:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Neo4j9527")

def check_neo4j_connection(uri, user, password):
    print(f"🔍 尝试连接 Neo4j: {uri}")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=10)
        with driver.session() as session:
            result = session.run("RETURN 'OK' AS msg")
            msg = result.single()["msg"]
            print(f"✅ 连接成功！服务正常响应: {msg}")
            print(f"   URI: {uri}")
            print(f"   用户: {user}")
            return True
    except ServiceUnavailable as e:
        print("❌ 连接失败：Neo4j 服务不可用")
        print("   可能原因：")
        print("   - Neo4j 未启动（请运行 `bin/neo4j console`）")
        print("   - 防火墙/安全组未开放 7687 端口")
        print("   - server.bolt.address 未设为 0.0.0.0（外部访问时）")
        print(f"   错误详情: {e}")
        return False
    except AuthError as e:
        print("❌ 认证失败：用户名或密码错误")
        print(f"   当前配置: user='{user}', password='{password}'")
        print("   请检查：")
        print("   - 是否首次登录 Web 界面设置了新密码？")
        print("   - 环境变量 NEO4J_PASSWORD 是否正确？")
        print(f"   错误详情: {e}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    success = check_neo4j_connection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    sys.exit(0 if success else 1)