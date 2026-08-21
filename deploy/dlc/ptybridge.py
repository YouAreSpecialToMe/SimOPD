#!/usr/bin/env python3
"""文件 PTY 桥:经共享盘把 pod 上的交互 shell 递到跳板机 —— 不要网络出站,不要凭据。

调试通道(task.sh sh)证明了"文件当信道"这条路是通的,但它一次一条命令、没有 tty:
vim/top/python -i 用不了,cd 也不跨命令保留。这个桥把它升级成真终端:pod 侧在 PTY 里
fork 一个 bash,PTY 的输出追加进会话目录的 o 文件、输入从 i 文件尾部读;跳板机侧把
本地终端置为 raw,键击追加进 i、盯着 o 把新字节写回屏幕。字节级透传,所以全屏程序、
控制序列、颜色都原样工作 —— 快不快只取决于共享盘跨节点的可见延迟。

用法(一般不直接调,走 task.sh tty <槽>):
    python3 ptybridge.py serve  <会话目录>     # pod 侧,由调试通道 setsid 拉起
    python3 ptybridge.py attach <会话目录>     # 跳板机侧,Ctrl-] 断开

会话目录协议(都在共享盘上):
    i      客户端 -> shell 的字节流(只追加)
    o      shell -> 客户端的字节流(只追加)
    winsz  "rows cols",客户端写、服务端轮询后 TIOCSWINSZ + SIGWINCH
    shb    服务端心跳(mtime);chb 客户端心跳
    rc     shell 退出码,出现即会话结束(最后写)
一场会话一个新目录、一个新 bash:这是"临时 shell",不是 tmux —— 断开即弃,
留会话保活/重连那套复杂度不值得在 v1 背。
"""
import os
import select
import signal
import sys
import time

IDLE_S = int(os.environ.get("PTY_IDLE_S", "1800"))       # 无任何活动多久后自杀
CHB_GRACE_S = int(os.environ.get("PTY_CHB_GRACE_S", "120"))  # 等第一个客户端多久
CHB_STALE_S = int(os.environ.get("PTY_CHB_STALE_S", "30"))   # 客户端心跳断多久算走了
O_CAP = int(os.environ.get("PTY_O_CAP_MB", "256")) * 1024 * 1024
DETACH = b"\x1d"                                          # Ctrl-]


def _size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def _mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0


def serve(d):
    import fcntl
    import pty
    import struct
    import termios

    os.makedirs(d, exist_ok=True)
    ipath, opath, rcpath = os.path.join(d, "i"), os.path.join(d, "o"), os.path.join(d, "rc")
    open(ipath, "ab").close()
    pid, master = pty.fork()
    if pid == 0:
        os.environ.setdefault("TERM", "xterm-256color")
        os.execvp("bash", ["bash", "-i"])
    of = open(opath, "ab", buffering=0)
    ioff = _size(ipath)          # 从现在起:目录复用时不吃前一场的旧输入
    t0 = time.time()
    last_act = t0
    last_hb = 0.0
    last_winsz = 0.0
    chb_seen = False
    shb = os.path.join(d, "shb")
    try:
        while True:
            now = time.time()
            if now - last_hb >= 2:
                last_hb = now
                with open(shb, "w") as f:
                    f.write(str(os.getpid()))
            r, _, _ = select.select([master], [], [], 0.05)
            if master in r:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    break                                 # shell 退出,PTY 关闭
                if not data:
                    break
                of.write(data)
                last_act = now
                if _size(opath) > O_CAP:
                    of.write("\r\n[ptybridge] 输出超过上限,会话结束\r\n".encode())
                    break
            sz = _size(ipath)
            if sz > ioff:
                with open(ipath, "rb") as f:
                    f.seek(ioff)
                    data = f.read(sz - ioff)
                ioff = sz
                if data:
                    os.write(master, data)
                    last_act = now
            wz = os.path.join(d, "winsz")
            m = _mtime(wz)
            if m > last_winsz:
                last_winsz = m
                try:
                    rows, cols = open(wz).read().split()[:2]
                    fcntl.ioctl(master, termios.TIOCSWINSZ,
                                struct.pack("HHHH", int(rows), int(cols), 0, 0))
                    os.kill(pid, signal.SIGWINCH)
                except (OSError, ValueError):
                    pass
            cm = _mtime(os.path.join(d, "chb"))
            if cm:
                chb_seen = True
            if chb_seen and now - cm > CHB_STALE_S:
                break                                     # 客户端走了(Ctrl-] 或断网)
            if not chb_seen and now - t0 > CHB_GRACE_S:
                break                                     # 从没人来接
            if now - last_act > IDLE_S:
                break
    finally:
        try:
            os.kill(pid, signal.SIGHUP)
            time.sleep(0.2)
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        rc = 0
        try:
            _, st = os.waitpid(pid, os.WNOHANG)
            if os.WIFEXITED(st):
                rc = os.WEXITSTATUS(st)
            elif os.WIFSIGNALED(st):
                rc = 128 + os.WTERMSIG(st)      # waitstatus_to_exitcode 是 3.9+,手拆兼容老 python
        except OSError:
            pass
        of.close()
        with open(rcpath, "w") as f:                      # 最后写 = 会话结束信号
            f.write(str(rc))


def attach(d):
    import termios
    import tty as ttymod

    ipath, opath, rcpath = os.path.join(d, "i"), os.path.join(d, "o"), os.path.join(d, "rc")
    chb = os.path.join(d, "chb")
    ifile = open(ipath, "ab", buffering=0)
    ooff = _size(opath)                                   # 只看新输出
    fd = sys.stdin.fileno()

    def push_winsz(*_):
        try:
            cols, rows = os.get_terminal_size()
            with open(os.path.join(d, "winsz"), "w") as f:
                f.write("%d %d" % (rows, cols))
        except OSError:
            pass

    push_winsz()
    signal.signal(signal.SIGWINCH, push_winsz)
    sys.stderr.write("[ptybridge] 已连接,Ctrl-] 断开(断开即弃,这不是 tmux)\r\n")
    old = termios.tcgetattr(fd)
    ttymod.setraw(fd)
    last_hb = 0.0
    try:
        while True:
            now = time.time()
            if now - last_hb >= 2:
                last_hb = now
                with open(chb, "w") as f:
                    f.write("1")
            r, _, _ = select.select([fd], [], [], 0.03)
            if fd in r:
                data = os.read(fd, 4096)
                if DETACH in data:
                    ifile.write(data.split(DETACH)[0])
                    break
                ifile.write(data)
            sz = _size(opath)
            if sz > ooff:
                with open(opath, "rb") as f:
                    f.seek(ooff)
                    data = f.read(sz - ooff)
                ooff = sz
                os.write(1, data)
            if os.path.exists(rcpath):
                sz = _size(opath)                         # 补吐 rc 出现前的尾巴
                if sz > ooff:
                    with open(opath, "rb") as f:
                        f.seek(ooff)
                        os.write(1, f.read(sz - ooff))
                sys.stderr.write("\r\n[ptybridge] shell 已退出(rc=%s)\r\n"
                                 % open(rcpath).read().strip())
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("serve", "attach"):
        sys.stderr.write(__doc__)
        return 2
    (serve if sys.argv[1] == "serve" else attach)(sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
