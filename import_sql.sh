#!/bin/bash

#################################################################
#  import_sql.sh
#  用法:
#       ./import_sql.sh XXX.sql
#
#  作用:
#       将指定 SQL 文件导入 nerdctl 运行的 MariaDB 容器
#################################################################

# === 1. 检查参数 ===
if [ $# -ne 1 ]; then
    echo "❌ 使用方式: $0 <sql-file>"
    exit 1
fi

SQL_FILE="$1"

# === 2. 检查文件是否存在 ===
if [ ! -f "$SQL_FILE" ]; then
    echo "❌ SQL 文件不存在: $SQL_FILE"
    exit 1
fi

# === 3. 检查 MariaDB 容器是否存在 ===
CONTAINER_NAME="mariadb"

if ! nerdctl ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ 找不到容器: ${CONTAINER_NAME}"
    echo "请先启动容器，例如："
    echo "  nerdctl run -d --name mariadb -e MYSQL_ROOT_PASSWORD=123456 -p 3306:3306 docker.m.daocloud.io/library/mariadb:10.2"
    exit 1
fi

echo "🔍 检查容器状态..."
# === 4. 检查容器是否正在运行 ===
if ! nerdctl ps | grep -q "${CONTAINER_NAME}"; then
    echo "❌ 容器已停止，请启动："
    echo "  nerdctl start mariadb"
    exit 1
fi

echo "📦 容器已运行：${CONTAINER_NAME}"

# === 5. 将 SQL 文件复制到容器中 ===
echo "📤 复制 SQL 文件到容器..."
nerdctl cp "$SQL_FILE" ${CONTAINER_NAME}:/tmp/import.sql

# === 6. 执行导入 ===
echo "📥 开始导入 SQL 数据..."

nerdctl exec -i ${CONTAINER_NAME} \
    sh -c "mysql -uroot -p123456 < /tmp/import.sql"

if [ $? -eq 0 ]; then
    echo "🎉 SQL 导入成功!"
else
    echo "❌ SQL 导入失败，请检查 SQL 文件内容或数据库状态"
    exit 1
fi