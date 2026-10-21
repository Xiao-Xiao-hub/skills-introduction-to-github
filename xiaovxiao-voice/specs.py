"""哓v哓 的声音设定 —— 想调整方向，改这个文件就行。

音色描述由「槽位」拼出来：年龄感、音调、质感、鼻音、语速、情绪底色、说话方式。
海选时用 ARCHETYPES 里的 10 个组合铺开试；精调时在最喜欢的组合附近，每次只换一个槽位。
"""

# 名字读 xiǎo xiǎo（三声），v 不读出来。「哓」是生僻字，模型会按字典读成一声，
# 所以台词里写同音的「晓晓」。
NAME_SPOKEN = "晓晓"

# 海选台词：所有候选读同一句，才能公平比较。约 10~12 秒，包含自我介绍、可爱、忧郁、停顿。
AUDITION_LINE = (
    f"大家好……我是{NAME_SPOKEN}。嗯，今天也有在好好生活哦。"
    "其实下雨天也没那么讨厌啦，只是……会有一点点想念某个人而已。"
)

# 每个槽位的可选值：键 -> (简称, 中文描述片段, 英文描述片段)。空字符串表示不写这一项。
SLOTS = {
    "age": {
        "15": ("15-16岁", "十五六岁的少女", "a girl about 15 or 16 years old"),
        "17": ("17-18岁", "十七八岁的少女", "a girl about 17 or 18 years old"),
        "20": ("20岁左右", "二十岁左右的年轻女生", "a young woman around 20"),
    },
    "pitch": {
        "mid": ("中等", "音调中等", "medium pitch"),
        "midhigh": ("中等偏高", "音调中等偏高", "medium-high pitch"),
        "high": ("偏高", "音调偏高", "high pitch"),
    },
    "texture": {
        "clear": ("清透", "声音清透干净", "a clear and clean voice"),
        "soft": ("柔软微气声", "声音柔软，带一点气声", "a soft voice with a little breathiness"),
        "airy": ("明显气声", "气声明显，声音轻飘飘的", "a noticeably airy, breathy and light voice"),
        "husky": ("柔软微沙", "声音柔软，略带一点沙哑", "a soft voice with a slightly husky texture"),
    },
    "nasal": {
        "none": ("无", "", ""),
        "light": ("轻微鼻音", "带一点轻微的鼻音", "a slight nasal tone"),
    },
    "pace": {
        "slow": ("慢", "语速慢，停顿较多", "a slow pace with frequent pauses"),
        "midslow": ("偏慢", "语速偏慢", "a slightly slow pace"),
        "normal": ("自然", "语速自然", "a natural speaking pace"),
    },
    "mood": {
        "melancholy": ("淡淡忧郁", "温柔里透着淡淡的忧郁", "gentle, with a faint melancholy"),
        "sleepy": ("慵懒困倦", "慵懒，像刚睡醒一样", "lazy and drowsy, as if she just woke up"),
        "lonely": ("安静寂寞", "语气平淡安静，有一点寂寞和疏离感", "calm and quiet, a little lonely and distant"),
        "sadsweet": ("甜而低落", "甜美但情绪有些低落，带一点委屈", "sweet but a little down, slightly aggrieved"),
        "shy": ("害羞", "害羞腼腆，有点不好意思", "shy and bashful"),
    },
    "manner": {
        "plain": ("不指定", "", ""),
        "whisper": ("轻声细语", "轻声细语", "soft-spoken, close to a whisper"),
        "murmur": ("自言自语", "像在小声自言自语", "as if murmuring to herself"),
        "smile": ("带笑意", "说话时带着浅浅的笑意", "with a faint smile in her voice"),
        "trail": ("句尾下沉", "句尾轻轻下沉", "letting sentence endings fall softly"),
    },
}

SLOT_ORDER = ["age", "pitch", "texture", "nasal", "pace", "mood", "manner"]
SLOT_LABELS = {
    "age": "年龄感", "pitch": "音调", "texture": "质感", "nasal": "鼻音",
    "pace": "语速", "mood": "情绪底色", "manner": "说话方式",
}

# 每条描述末尾都加上，压一压「夹」和「做作」
DESC_SUFFIX = ("声音自然真实，不做作", "natural and unaffected, not exaggerated")

# 海选的 10 个方向（代号, 名字, 槽位）。前几个最贴参考图，后面的用来探索边界。
ARCHETYPES = [
    ("A", "软糯忧郁", dict(age="17", pitch="midhigh", texture="soft", nasal="light", pace="midslow", mood="melancholy", manner="whisper")),
    ("B", "清冷慵懒", dict(age="17", pitch="mid", texture="clear", nasal="none", pace="slow", mood="lonely", manner="plain")),
    ("C", "甜而低落", dict(age="15", pitch="high", texture="soft", nasal="light", pace="midslow", mood="sadsweet", manner="murmur")),
    ("D", "困倦小猫", dict(age="17", pitch="midhigh", texture="airy", nasal="light", pace="slow", mood="sleepy", manner="smile")),
    ("E", "温柔学姐", dict(age="20", pitch="mid", texture="soft", nasal="none", pace="midslow", mood="melancholy", manner="smile")),
    ("F", "害羞内向", dict(age="15", pitch="midhigh", texture="clear", nasal="none", pace="midslow", mood="shy", manner="whisper")),
    ("G", "雨天独白", dict(age="17", pitch="mid", texture="husky", nasal="none", pace="slow", mood="melancholy", manner="trail")),
    ("H", "轻飘飘", dict(age="17", pitch="high", texture="airy", nasal="none", pace="slow", mood="lonely", manner="whisper")),
    ("I", "笑着藏心事", dict(age="17", pitch="midhigh", texture="clear", nasal="light", pace="normal", mood="melancholy", manner="smile")),
    ("J", "软萌低语", dict(age="15", pitch="midhigh", texture="soft", nasal="light", pace="slow", mood="sleepy", manner="murmur")),
]

# 克隆测试用的台词：(标签, 语气控制, 台词)。语气控制只有 VoxCPM2 支持，留空则用高保真克隆。
# 数字写成汉字，避免读法不稳定。
CLONE_TEST_LINES = [
    ("日常", "", "今天的视频会有一点长，记得泡一杯热可可，慢慢看就好。"),
    ("忧郁", "语气低落伤感，声音轻轻的", "如果哪天我不见了，大概只是躲到哪个角落里，偷偷睡着了吧。"),
    ("可爱", "语气开心俏皮，带着笑意", "诶嘿，被你发现啦？那、那我把面具戴上好了！"),
    ("疑问", "", "你也会在半夜突然觉得，有点想哭吗？"),
    ("长句", "", "其实我一直觉得，这个世界上最温柔的东西，是傍晚的风、刚晒过的被子，还有一个愿意听你说废话的人。"),
    ("中英混", "", "下一期我们来聊聊今年最喜欢的十首歌，里面还有一首 English song 哦。"),
]

# 试音台里的标签 -> 精调时建议优先听哪个槽位往哪边变（只用于提示）
TAG_HINTS = {
    "太成熟": ("age", "更小"),
    "太幼": ("age", "更大"),
    "太尖": ("pitch", "更低"),
    "太闷": ("pitch", "更高"),
    "夹/做作": ("texture", "更清透"),
    "不够忧郁": ("mood", "淡淡忧郁 / 安静寂寞"),
    "不够可爱": ("manner", "带笑意"),
    "哭腔重": ("mood", "慵懒困倦 / 安静寂寞"),
}
