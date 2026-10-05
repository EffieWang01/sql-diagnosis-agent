"""沙箱安全性测试 —— 这是整个项目最不能出错的地方。"""

from __future__ import annotations

from sqlagent.env.sandbox import ERR_ILLEGAL, ERR_TIMEOUT, SQLSandbox
from sqlagent.config import SandboxConfig


class TestStatementGuard:
    """语句白名单。"""

    def test_select_allowed(self, sandbox):
        r = sandbox.run("SELECT COUNT(*) FROM sales")
        assert r.ok, r.error

    def test_with_allowed(self, sandbox):
        r = sandbox.run("WITH t AS (SELECT 1 AS x) SELECT * FROM t")
        assert r.ok, r.error

    def test_drop_rejected(self, sandbox):
        r = sandbox.run("DROP TABLE sales")
        assert not r.ok
        assert r.error_type == ERR_ILLEGAL

    def test_insert_rejected(self, sandbox):
        r = sandbox.run("INSERT INTO sales VALUES (9,'2026-01-01','华东','手机',1)")
        assert not r.ok
        assert r.error_type == ERR_ILLEGAL

    def test_update_rejected(self, sandbox):
        r = sandbox.run("UPDATE sales SET gmv = 0")
        assert not r.ok and r.error_type == ERR_ILLEGAL

    def test_attach_rejected(self, sandbox):
        r = sandbox.run("ATTACH 'other.db' AS o")
        assert not r.ok and r.error_type == ERR_ILLEGAL

    def test_copy_rejected(self, sandbox):
        r = sandbox.run("COPY sales TO '/tmp/out.csv'")
        assert not r.ok and r.error_type == ERR_ILLEGAL

    def test_multiple_statements_rejected(self, sandbox):
        r = sandbox.run("SELECT 1; DROP TABLE sales")
        assert not r.ok and r.error_type == ERR_ILLEGAL

    def test_read_csv_rejected(self, sandbox):
        r = sandbox.run("SELECT * FROM read_csv('/etc/passwd')")
        assert not r.ok and r.error_type == ERR_ILLEGAL

    def test_empty_rejected(self, sandbox):
        assert not sandbox.run("   ").ok

    def test_data_still_intact_after_attacks(self, sandbox):
        """所有攻击之后数据必须原样存在。"""
        for bad in ("DROP TABLE sales", "DELETE FROM sales", "UPDATE sales SET gmv=0"):
            sandbox.run(bad)
        r = sandbox.run("SELECT COUNT(*) FROM sales")
        assert r.ok and r.rows[0][0] == 5


class TestKeywordFalsePositives:
    """关键字检查不能误伤正常 SQL。"""

    def test_column_named_created_at(self, tmp_db):
        import duckdb

        con = duckdb.connect(str(tmp_db))
        con.execute("CREATE TABLE t2 (created_at DATE, updated_at DATE, offset_v INT)")
        con.execute("INSERT INTO t2 VALUES ('2026-01-01','2026-01-02', 3)")
        con.close()
        sb = SQLSandbox(tmp_db, SandboxConfig())
        r = sb.run("SELECT created_at, updated_at, offset_v FROM t2")
        sb.close()
        assert r.ok, f"被误判为违规: {r.error}"

    def test_string_literal_containing_keyword(self, sandbox):
        r = sandbox.run("SELECT 'DROP TABLE' AS note")
        assert r.ok, f"字符串字面量里的关键字被误判: {r.error}"

    def test_string_literal_containing_function_name(self, sandbox):
        r = sandbox.run("SELECT 'read_csv is dangerous' AS note")
        assert r.ok, f"字符串里的函数名被误判: {r.error}"


class TestResultLimits:
    def test_row_truncation(self, sandbox):
        r = sandbox.run("SELECT * FROM sales")
        assert r.ok
        assert len(r.rows) == 3, "应被 max_rows=3 截断"
        assert r.truncated is True

    def test_not_truncated_when_small(self, sandbox):
        r = sandbox.run("SELECT * FROM sales LIMIT 2")
        assert r.ok and r.truncated is False and len(r.rows) == 2

    def test_observation_mentions_truncation(self, sandbox):
        r = sandbox.run("SELECT * FROM sales")
        assert "截断" in r.to_observation()


class TestErrors:
    def test_execution_error_classified(self, sandbox):
        r = sandbox.run("SELECT * FROM 不存在的表")
        assert not r.ok
        assert r.error_type == "execution"
        assert r.error  # 有可读的错误信息

    def test_run_never_raises(self, sandbox):
        """run() 必须永不抛异常——轨迹不能因为一条坏 SQL 就崩掉。"""
        for sql in ["", "DROP TABLE x", "SELECT * FROM nope", "!!!", "SELECT 1; SELECT 2"]:
            r = sandbox.run(sql)
            assert r is not None

    def test_timeout(self, tmp_db):
        sb = SQLSandbox(tmp_db, SandboxConfig(timeout_s=0.4, max_rows=10))
        # 笛卡尔积自连接，保证跑够久
        r = sb.run(
            "SELECT COUNT(*) FROM range(20000) a, range(20000) b, range(400) c"
        )
        sb.close()
        assert not r.ok
        assert r.error_type == ERR_TIMEOUT
        assert "超时" in r.error


class TestStats:
    def test_counters(self, sandbox):
        sandbox.run("SELECT 1")
        sandbox.run("DROP TABLE sales")
        sandbox.run("SELECT * FROM nope")
        assert sandbox.stats["ok"] == 1
        assert sandbox.stats["illegal"] == 1
        assert sandbox.stats["execution_error"] == 1
        assert sandbox.stats["total"] == 3

    def test_read_only_flag(self, tmp_db):
        sb = SQLSandbox(tmp_db, SandboxConfig())
        assert sb.read_only is True
        sb.close()

    def test_external_access_disabled(self, tmp_db):
        sb = SQLSandbox(tmp_db, SandboxConfig())
        # 是否成功关闭取决于 DuckDB 版本，但不应抛异常
        assert isinstance(sb.external_access_disabled, bool)
        sb.close()
