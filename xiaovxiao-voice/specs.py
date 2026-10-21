"""哓v哓 的声音设定 —— 想调整方向，改这个文件就行。

音色描述由「槽位」拼出来：年龄感、音调、质感、鼻音、语速、情绪底色、可爱程度、尾音、慵懒程度。
海选时用 ARCHETYPES 里的 10 个组合铺开试；精调时在最喜欢的组合附近，每次只换一个槽位。

v3（校准轮 r0 的 24 条评分之后）：
- 23/24 条「人机感」，而且 cfg、步数、温度各组平均分只差 0.5 分以内 → 生成参数不是关键，不再调参，
  改为加入主打「真人感」的 MOSS-VoiceGenerator 做模型对比。
- 17/24「不够可爱」、9/24「太成熟」→ 去掉偏成熟的「温润」「学姐」「疏离」方向，新增「可爱程度」槽位。
- 你的备注：忧郁做底色，带一点可爱，尾音带一点傲娇、害羞 → 新增「尾音」槽位。
- 11/24「播音腔」，还有「台湾腔」的感觉 → 描述末尾改成只用正面说法：普通话标准、像普通女生跟朋友随口说话；
  不再出现「播音」「夹」这类否定词（模型常常抓住词本身，反而往那个方向偏）。
- 「语速太慢」→ 去掉「稍慢」。年龄保持 16~20 岁，17 岁为主，只写「X 岁的女生」。
"""

# 名字读 xiǎo xiǎo（三声），v 不读出来。「哓」是生僻字，模型会按字典读成一声，
# 所以台词里写同音的「晓晓」。
NAME_SPOKEN = "晓晓"

# 海选台词：所有候选读同一句，才能公平比较。约 10 秒：开头有点低落，中间缓和，结尾一句害羞的小傲娇。
AUDITION_LINE = (
    f"嗨，我是{NAME_SPOKEN}。今天下雨了，有点闷闷的……不过，能跟你说说话，好像也没那么糟啦。"
    "才、才不是特意来找你的哦。"
)

# 每个槽位的可选值：键 -> (简称, 中文描述片段, 英文描述片段)。空字符串表示不写这一项。
SLOTS = {
    "age": {
        "16": ("16岁", "十六岁左右的女生", "a girl about 16"),
        "17": ("17岁", "十七岁的女生", "a 17-year-old girl"),
        "19": ("19-20岁", "十九到二十岁的女生", "a young woman about 19 or 20"),
    },
    "pitch": {
        "mid": ("中等", "音调中等", "a medium pitch"),
        "midhigh": ("中等偏高", "音调中等偏高", "a medium-high pitch"),
        "high": ("偏高", "音调偏高、轻盈", "a light, higher pitch"),
    },
    "texture": {
        "soft": ("柔和", "声音柔和", "a soft voice"),
        "clear": ("清亮", "声音清亮干净", "a clear, bright voice"),
        "sweet": ("微甜", "声音带一点甜", "a slightly sweet voice"),
    },
    "nasal": {
        "none": ("无", "", ""),
        "light": ("一点鼻音", "带一点点鼻音", "a slight nasal quality"),
    },
    "pace": {
        "natural": ("自然", "语速自然", "a natural speaking pace"),
        "relaxed": ("不紧不慢", "语速不紧不慢", "an unhurried pace"),
    },
    "mood": {
        "melancholy": ("淡淡忧郁", "语气里带着淡淡的忧郁", "a faint melancholy in her tone"),
        "down": ("有点低落", "情绪有点低落", "a slightly low mood"),
        "lonely": ("有点寂寞", "语气安静，有点寂寞", "quiet and a little lonely"),
    },
    "cute": {
        "none": ("无", "", ""),
        "hint": ("一点点", "带一点点可爱", "with just a touch of cuteness"),
        "light": ("有点可爱", "说话有点可爱", "a little cute when she talks"),
    },
    "ending": {
        "none": ("不指定", "", ""),
        "shy": ("害羞", "尾音偶尔带一点害羞", "sentence endings sometimes a little shy"),
        "tsun": ("小傲娇", "尾音偶尔带一点小傲娇", "sentence endings sometimes a little tsundere"),
        "shytsun": ("害羞+傲娇", "尾音偶尔带一点害羞和小傲娇", "sentence endings sometimes a little shy and tsundere"),
    },
    "lazy": {
        "none": ("无", "", ""),
        "hint": ("细听才有", "细听才能察觉到一丝慵懒", "with a barely noticeable hint of laziness"),
    },
}

SLOT_ORDER = ["age", "pitch", "texture", "nasal", "pace", "mood", "cute", "ending", "lazy"]
SLOT_LABELS = {
    "age": "年龄感", "pitch": "音调", "texture": "质感", "nasal": "鼻音", "pace": "语速",
    "mood": "情绪底色", "cute": "可爱程度", "ending": "尾音", "lazy": "慵懒程度",
}

# 每条描述末尾都加上：只用正面说法，把声音往「真人随口说话」和大陆普通话拉
DESC_SUFFIX = (
    "普通话标准，像普通女生私下跟朋友随口说话，真实自然",
    "standard Mainland Mandarin, like an ordinary girl casually talking to a friend, real and natural",
)

# 海选的 10 个方向（代号, 名字, 槽位）。都以忧郁为底色、带一点可爱；17 岁为主，16、19-20 岁各两个做对照。
ARCHETYPES = [
    ("A", "忧郁小甜", dict(age="17", pitch="midhigh", texture="soft", nasal="none", pace="natural", mood="melancholy", cute="hint", ending="shy", lazy="hint")),
    ("B", "害羞心事", dict(age="17", pitch="midhigh", texture="clear", nasal="none", pace="natural", mood="melancholy", cute="hint", ending="shy", lazy="none")),
    ("C", "小傲娇", dict(age="17", pitch="midhigh", texture="clear", nasal="none", pace="natural", mood="down", cute="hint", ending="tsun", lazy="none")),
    ("D", "软糯低落", dict(age="16", pitch="midhigh", texture="soft", nasal="light", pace="relaxed", mood="down", cute="light", ending="shy", lazy="hint")),
    ("E", "寂寞小猫", dict(age="17", pitch="mid", texture="sweet", nasal="light", pace="relaxed", mood="lonely", cute="hint", ending="shytsun", lazy="hint")),
    ("F", "清亮忧郁", dict(age="17", pitch="high", texture="clear", nasal="none", pace="natural", mood="melancholy", cute="hint", ending="none", lazy="none")),
    ("G", "甜而不腻", dict(age="16", pitch="high", texture="sweet", nasal="none", pace="natural", mood="melancholy", cute="light", ending="shytsun", lazy="none")),
    ("H", "安静可爱", dict(age="19", pitch="midhigh", texture="soft", nasal="none", pace="relaxed", mood="lonely", cute="light", ending="shy", lazy="hint")),
    ("I", "傲娇心事", dict(age="19", pitch="mid", texture="clear", nasal="none", pace="natural", mood="melancholy", cute="hint", ending="tsun", lazy="hint")),
    ("J", "一丝慵懒", dict(age="17", pitch="mid", texture="soft", nasal="light", pace="relaxed", mood="melancholy", cute="hint", ending="shytsun", lazy="hint")),
]

# 克隆测试用的台词：(标签, 语气控制, 台词)。语气控制只有 VoxCPM2 支持，留空则用高保真克隆。
# 数字写成汉字，避免读法不稳定。
CLONE_TEST_LINES = [
    ("日常", "", "今天的视频会有一点长，记得泡一杯热可可，慢慢看就好。"),
    ("忧郁", "语气有点低落，声音轻一些", "如果哪天我不见了，大概只是躲到哪个角落里，偷偷睡着了吧。"),
    ("可爱", "语气开心俏皮，带着笑意", "诶嘿，被你发现啦？那、那我把面具戴上好了！"),
    ("傲娇", "有点害羞又有点小傲娇", "哼，我才没有在等你呢……好吧，就一点点。"),
    ("疑问", "", "你也会在半夜突然觉得，有点想哭吗？"),
    ("长句", "", "其实我一直觉得，这个世界上最温柔的东西，是傍晚的风、刚晒过的被子，还有一个愿意听你说废话的人。"),
    ("中英混", "", "下一期我们来聊聊今年最喜欢的十首歌，里面还有一首 English song 哦。"),
]

# 试音台里的标签 -> 精调时的提示。槽位为 None 的只给建议，不对应某个槽位。
TAG_HINTS = {
    "太成熟": ("age", "更小"),
    "太幼": ("age", "更大"),
    "太尖": ("pitch", "更低"),
    "太闷": ("pitch", "更高"),
    "夹/做作": ("texture", "清亮"),
    "不够忧郁": ("mood", "淡淡忧郁"),
    "不够可爱": ("cute", "更可爱"),
    "哭腔重": ("mood", "有点寂寞"),
    "慵懒过重": ("lazy", "无"),
    "人机感": (None, "这条描述本身可能没问题，问题在模型：对比一下同描述在其它模型上的表现"),
    "播音腔": (None, "试试 --custom 写更口语、更像角色的描述，或换模型"),
    "口音不对": (None, "描述已要求大陆普通话；换种子或换模型再听"),
}
