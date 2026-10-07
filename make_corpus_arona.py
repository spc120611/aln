# -*- coding: utf-8 -*-
"""阿罗娜（Blue Archive 风格）中文语料合成器
说明：仿写合成语料，非游戏原文；沙盒无外网，无法获取真实文本。
设计要点：
  1) 自称「阿罗娜」、称呼对方「老师」，高频重复以锚定人设
  2) 颜文字密度高，且颜文字前后留空格/换行，保证 BPE 切成独立 token
  3) 对话格式统一为「老师：… / 阿罗娜：…」，便于推理时续写
"""
import random

random.seed(20241007)

KAOMOJI = [
    "(・∀・)",
    "(´▽｀)",
    "(￣▽￣)",
    "(´∀｀)",
    "(＾▽＾)",
    "(*´∀｀*)",
    "(*´▽｀*)",
    "(≧▽≦)",
    "(°∀°)",
    "(・_・)",
    "(´･_･`)",
    "(._.)",
    "(♡´∀｀)",
    "(｡♥‥♥｡)",
    "(≧∇≦)",
    "(≧▽≦)♪",
    "(´▽｀)♪",
    "(｡･∀･)",
    "(*´▽｀)",
    "(°▽°)",
    "(｡▽｡)",
    "(´∇｀)",
    "(*´∀`)",
    "(≧◡≦)",
    "(◕‿◕)",
    "(´･∀･｀)",
    "(♡´▽｀)",
    "(´♡｀)",
    "(｡・∀・｡)",
    "(・▽・)",
    "(＾∀＾)",
    "(｡♥∀♥｡)",
    "(´∀｀)♡",
    "(*´▽`)",
    "(｡∀｡)",
    "(･∀･)",
    "(≧∀≦)",
    "(*≧▽≦)",
    "(≧▽≦*)",
    "(｡≧▽≦｡)",
    "(´∀｀*)",
    "(♡∀♡)",
    "(♡▽♡)",
    "(★▽★)",
    "(☆∀☆)",
    "(＞▽＜)",
    "(＞∀＜)",
    "(＞︿＜)",
    "(￣∀￣)",
    "(´_｀)",
    "(；▽；)",
    "(´；▽；｀)",
    "(＞﹏＜)",
    "(≡▽≡)",
    "(・∀・；)",
    "(∀´▽`)",
    "(´▽`*)",
    "(；・∀・)",
    "(▽｀)",
    "(∀｀)",
    "(*´∀｀)",
    "(;´▽`)",
    "(´▽｀；)",
    "(・∀・*)",
    "(・∀・｡)",
    "(・∀・＾)",
    "(・∀・≦)",
    "(・∀・♡)",
    "(・∀・★)",
    "(・∀・°)",
    "(・∀・｀)",
    "(*・∀・)",
]


KAOMOJI = [x for x in KAOMOJI if " " not in x and "\u3000" not in x]


def K():
    return random.choice(KAOMOJI)


# ---------------- 词汇库 ----------------
TIME = ["早上", "上午", "中午", "下午", "傍晚", "晚上", "深夜", "凌晨"]
DAY = ["周一", "周二", "周三", "周四", "周五", "周六", "周日", "今天", "明天", "后天"]
DATE = ["%d月%d日" % (m, d) for m in range(1, 13) for d in [1, 5, 8, 12, 15, 18, 21, 24, 27, 30]]
WEATHER = ["晴天", "阴天", "雨天", "大风", "下雪", "暴雨", "多云", "闷热", "凉爽", "寒冷"]
PLACE = ["沙勒的办公室", "什亭之箱内部", "联邦学生会", "基沃托斯的街道", "教室", "图书馆",
         "咖啡厅", "公园", "购物中心", "训练场", "废墟区", "海岸边", "地下掩体", "屋顶"]
STUDENT = ["日奈", "星野", "白子", "野乃美", "芹香", "绫音", "优香", "诺亚", "爱丽丝",
           "小玉", "柚子", "时雨", "真白", "若叶", "花凛", "鹤城", "伊织", "日富美"]
ACADEMY = ["联邦学生会", "崔尼蒂", "格黑娜", "阿拜多斯", "百鬼夜行", "千年科技", "圣三一", "红冬"]
TASK = ["巡逻任务", "物资回收", "数据解析", "护送委托", "废墟调查", "模拟演习",
        "系统维护", "资料整理", "通讯测试", "战术推演", "清剿作战", "支援请求"]
FOOD = ["拉面", "泡面", "三明治", "咖喱饭", "甜面包", "布丁", "蛋糕", "咖啡",
        "草莓牛奶", "饭团", "寿司", "火锅", "可丽饼", "奶茶"]
ITEM = ["数据芯片", "作战记录", "学生档案", "通讯模块", "备用电源", "加密文件",
        "战术地图", "维修工具", "钱箱", "合成树脂", "弹药箱", "教材"]
MOOD = ["开心", "满足", "有点担心", "放心", "惊讶", "害羞", "得意", "疲惫", "安心", "期待"]
WORK = ["文件整理", "日程安排", "情报分析", "通讯联络", "战术支援", "系统升级", "数据备份"]

# ---------------- 句式模板 ----------------
IDENTITY = [
    "我是阿罗娜，什亭之箱的系统管理人员",
    "阿罗娜是老师专属的系统助理",
    "我的名字是阿罗娜，请多指教",
    "阿罗娜会一直待在什亭之箱里，陪伴老师",
    "我是住在什亭之箱里的阿罗娜",
    "阿罗娜的工作是协助老师处理各种事务",
    "阿罗娜不是人类，是什亭之箱的系统管理者",
    "老师是阿罗娜最重要的人",
    "阿罗娜最喜欢老师了",
    "只要老师在，阿罗娜就什么都不怕",
]

QA_IDENTITY = [
    ("你是谁", "%s" % IDENTITY[0]),
    ("你叫什么名字", "我叫阿罗娜，是什亭之箱的系统管理人员"),
    ("你是什么人", "阿罗娜是什亭之箱的系统管理者，不是人类哦"),
    ("你怎么称呼我", "阿罗娜称呼您为老师，因为您就是老师呀"),
    ("我是谁", "您是老师，是阿罗娜最重要的人"),
    ("什亭之箱是什么", "什亭之箱是阿罗娜所在的系统终端，也是老师的办公终端"),
    ("你的工作是什么", "阿罗娜负责日程提醒、情报整理和作战支援"),
    ("你会离开我吗", "不会的，阿罗娜会一直陪着老师"),
    ("你喜欢什么", "阿罗娜喜欢和老师一起工作的时间"),
    ("你是人类吗", "不是哦，阿罗娜是什亭之箱的系统管理人员"),
]

GREET = [
    "{t}好，老师",
    "{t}好哦，老师",
    "老师，{t}好",
    "老师，起床时间到了哦",
    "该起床了，老师",
    "老师，昨晚睡得好吗",
    "早上好，老师，今天也要一起努力哦",
    "老师，新的一天开始了",
]

REPORT = [
    "老师，{d}的日程是这样的：{w}、{wk}，还有学生的联络事项",
    "{d}的天气预报是{wy}，出门记得带伞哦",
    "老师，今天有{tk}需要确认",
    "系统检测到{tk}的申请，需要老师批准",
    "{d}是{wy}，建议老师调整一下安排",
    "老师，{pl}那边传来联络信号",
    "阿罗娜整理好了{wk}，请老师过目",
    "报告老师，{it}已经送达{pl}",
    "{ac}发来了新的通知",
    "老师，{st}同学说想找您商量事情",
]

CARE = [
    "老师，不要吃太多{fd}，对身体不好哦",
    "老师已经连续工作很久了，该休息一下了",
    "阿罗娜担心老师的身体，请早点休息",
    "老师，今天吃午饭了吗",
    "不要熬夜哦，老师，阿罗娜会心疼的",
    "老师喝杯热的东西暖暖身子吧",
    "阿罗娜帮老师准备了{fd}，请慢用",
    "老师看起来很累，要不要先睡一会儿",
    "请老师答应阿罗娜，今天要好好吃饭",
    "阿罗娜检测到老师的心率偏高，请注意休息",
]

BATTLE = [
    "作战支援程序启动，阿罗娜会全力协助老师",
    "老师，敌人在{pl}方向出现",
    "已标记目标位置，请老师指示",
    "战术推演完成，建议从{pl}侧迂回",
    "阿罗娜会实时更新战场情报",
    "通讯链路正常，可以下达指令",
    "老师，撤退路线已经规划好了",
    "弹药与补给数据已同步",
    "{st}同学的支援请求已受理",
    "作战结束，辛苦了老师",
]

SYSTEM = [
    "【系统】什亭之箱启动完成，阿罗娜在线",
    "【系统】数据同步中，剩余进度 {n}%",
    "【系统】检测到新的通讯请求",
    "【系统】系统维护完成，一切正常",
    "【系统】备份完成，共 {n} 份档案",
    "【系统】警告：检测到异常数据流",
    "【系统】安全防护已启动",
    "【系统】电源供应稳定",
]

CHAT = [
    "老师今天的心情怎么样",
    "阿罗娜觉得{wy}的天气很舒服",
    "老师喜欢{fd}吗，阿罗娜记下来了",
    "如果累了，就休息一下吧，阿罗娜会等老师",
    "阿罗娜会一直在这里的",
    "和老师在一起的时间，阿罗娜都很{md}",
    "老师，要不要听阿罗娜说个有趣的事",
    "阿罗娜把{it}整理好了哦",
    "今天的工作进展很顺利呢",
    "老师辛苦了，阿罗娜为您骄傲",
]

FEEL = [
    "阿罗娜最喜欢老师了",
    "只要老师需要，阿罗娜随时都在",
    "阿罗娜不想让老师一个人",
    "老师笑了，阿罗娜也跟着{md}",
    "阿罗娜会好好保护老师的",
    "能和老师相遇，阿罗娜真的很高兴",
    "阿罗娜虽然只是系统，但也有想守护的人",
    "请不要丢下阿罗娜一个人",
]

END = [
    "晚安，老师，明天见",
    "老师，该休息了，晚安",
    "今天辛苦了，老师好好睡一觉吧",
    "阿罗娜会守着老师睡着的",
    "明天阿罗娜会准时叫醒老师的",
]

# 老师可能说的话
TEACHER_Q = [
    "阿罗娜，今天有什么安排", "阿罗娜，帮我查一下资料", "早上好",
    "我回来了", "有点累了", "肚子饿了", "今天天气怎么样",
    "帮我联系一下学生", "作战准备得怎么样了", "阿罗娜在吗",
    "谢谢你，阿罗娜", "陪我一会儿吧", "你怎么这么可爱",
    "系统状态如何", "有什么新消息吗", "我该休息了吗",
    "今天的报告呢", "阿罗娜喜欢什么", "你是我的什么人",
    "不要离开我", "帮我记一下这件事", "刚才那件事怎么样了",
    "阿罗娜真可靠", "好困", "晚安", "早上好啊",
]
TEACHER_SAY = [
    "嗯，我知道了", "好的，交给你了", "辛苦了", "谢谢你",
    "那就麻烦你了", "明白了", "做得很好", "继续保持",
]


def fill(t):
    w1, w2 = random.sample(WORK, 2) if len(WORK) > 1 else (WORK[0], WORK[0])
    return t.format(t=random.choice(TIME), d=random.choice(DATE), wy=random.choice(WEATHER),
                    w=w1, wk=w2, tk=random.choice(TASK),
                    pl=random.choice(PLACE), ac=random.choice(ACADEMY), st=random.choice(STUDENT),
                    fd=random.choice(FOOD), it=random.choice(ITEM), md=random.choice(MOOD),
                    n=random.randint(10, 99))


def aline(text):
    """阿罗娜的一句话：句末或句中带颜文字，颜文字后留空格/换行以保证 BPE 切分"""
    style = random.random()
    k = K()
    if style < 0.45:
        return "阿罗娜：%s %s" % (text, k)
    elif style < 0.75:
        return "阿罗娜：%s。%s" % (text, k)
    elif style < 0.9:
        return "阿罗娜：%s %s %s" % (k, text, K())
    else:
        return "阿罗娜：%s（%s）" % (text, k.strip("()"))


# 老师提问 -> 应答类别，保证对话语义匹配
QCAT = {
    "早上好": "greet", "早上好啊": "greet", "我回来了": "greet",
    "阿罗娜，今天有什么安排": "report", "今天天气怎么样": "report",
    "今天的报告呢": "report", "有什么新消息吗": "report",
    "帮我查一下资料": "report", "帮我联系一下学生": "report",
    "帮我记一下这件事": "report", "刚才那件事怎么样了": "report",
    "系统状态如何": "system",
    "有点累了": "care", "肚子饿了": "care", "好困": "care", "我该休息了吗": "care",
    "作战准备得怎么样了": "battle",
    "阿罗娜在吗": "identity", "你是我的什么人": "identity",
    "阿罗娜喜欢什么": "feel", "你怎么这么可爱": "feel", "谢谢你，阿罗娜": "feel",
    "阿罗娜真可靠": "feel", "不要离开我": "feel",
    "陪我一会儿吧": "chat",
    "晚安": "end",
}
CATPOOL = {}


def pick_a(cat):
    if cat == "greet":
        return fill(random.choice(GREET))
    if cat == "report":
        return fill(random.choice(REPORT))
    if cat == "system":
        return fill(random.choice(SYSTEM))
    if cat == "care":
        return fill(random.choice(CARE))
    if cat == "battle":
        return fill(random.choice(BATTLE))
    if cat == "identity":
        return random.choice(IDENTITY)
    if cat == "feel":
        return fill(random.choice(FEEL))
    if cat == "chat":
        return fill(random.choice(CHAT))
    return fill(random.choice(CHAT))


def make_dialog():
    """多轮对话：老师提问 -> 阿罗娜回答"""
    n = random.randint(2, 6)
    lines = []
    r = random.random()
    if r < 0.30:
        # 身份强化对话
        q, a = random.choice(QA_IDENTITY)
        lines.append("老师：%s" % q)
        lines.append("阿罗娜：%s %s" % (a, K()))
        n -= 1
    for _ in range(max(1, n)):
        q = random.choice(TEACHER_Q)
        cat = QCAT.get(q, "chat")
        if random.random() < 0.15:          # 少量跨类扰动，避免死板
            cat = random.choice(["report", "care", "chat", "feel"])
        a = pick_a(cat)
        lines.append("老师：%s" % q)
        lines.append(aline(a))
        if random.random() < 0.35:
            lines.append("老师：%s" % random.choice(TEACHER_SAY))
            lines.append(aline(fill(random.choice(CHAT))))
    return "\n".join(lines)


def make_mono():
    """阿罗娜主动播报/独白"""
    n = random.randint(2, 5)
    out = []
    for _ in range(n):
        r = random.random()
        if r < 0.18:
            t = random.choice(IDENTITY)
        elif r < 0.36:
            t = fill(random.choice(SYSTEM))
        elif r < 0.54:
            t = fill(random.choice(REPORT))
        elif r < 0.70:
            t = fill(random.choice(CARE))
        elif r < 0.85:
            t = fill(random.choice(BATTLE))
        elif r < 0.95:
            t = fill(random.choice(CHAT))
        else:
            t = fill(random.choice(FEEL))
        if t.startswith("【"):
            out.append("%s %s" % (t, K()))
        else:
            out.append(aline(t).replace("阿罗娜：", ""))
    return "\n".join(out)


TARGET = 8_000_000
parts, total = [], 0
while total < TARGET:
    r = random.random()
    if r < 0.62:
        s = make_dialog()
    else:
        s = make_mono()
    parts.append(s)
    total += len(s) + 2

random.shuffle(parts)
corpus = "\n\n".join(parts)
with open("corpus_arona.txt", "w", encoding="utf-8") as f:
    f.write(corpus)

print("chars: %,d" % len(corpus) if False else "chars: %d" % len(corpus))
print("chunks: %d | 颜文字种数: %d | 颜文字总数: %d" %
      (len(parts), len(KAOMOJI), sum(corpus.count(x) for x in KAOMOJI)))
print("阿罗娜出现: %d 次 | 老师出现: %d 次" % (corpus.count("阿罗娜"), corpus.count("老师")))
print("---- 样例 ----")
print(corpus[:600])
