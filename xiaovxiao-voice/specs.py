"""哓v哓 的声音设定 —— 想调整方向，改这个文件就行。

音色描述由「槽位」拼出来：年龄感、音调、质感、鼻音、语速、情绪底色、慵懒程度、说话方式。
海选时用 ARCHETYPES 里的 10 个组合铺开试；精调时在最喜欢的组合附近，每次只换一个槽位。

v2（第一次试听后）：
- 年龄定在 16~20 岁，17 岁为主。第一次偏幼主要是「少女」「轻声细语」这些词把声音往小、往夹带，
  所以年龄只写「X 岁的女生」，不写「少女」。
- 人机感重 → 去掉「气声明显」「语速慢、停顿多」「自言自语」等容易出 ASMR/机器腔的词，
  每条描述都要求「真人日常说话、口语自然、有语调起伏」；台词也改成口语，少用省略号。
- 慵懒要「细听才有」→ 慵懒单独做成一个槽位，默认用最轻的一档。
"""

# 名字读 xiǎo xiǎo（三声），v 不读出来。「哓」是生僻字，模型会按字典读成一声，
# 所以台词里写同音的「晓晓」。
NAME_SPOKEN = "晓晓"

# 海选台词：所有候选读同一句，才能公平比较。约 10 秒，像跟熟人随口说话。
AUDITION_LINE = (
    f"嗨，我是{NAME_SPOKEN}。嗯，今天也没什么特别的事，就是想来跟你说说话。"
    "其实下雨天也挺好的，只是偶尔……会有点想念某个人吧。"
)

# 每个槽位的可选值：键 -> (简称, 中文描述片段, 英文描述片段)。空字符串表示不写这一项。
SLOTS = {
    "age": {
        "16": ("16岁", "十六岁左右的女生", "a girl about 16"),
        "17": ("17岁", "十七岁的女生", "a 17-year-old girl"),
        "19": ("19-20岁", "十九到二十岁的女生", "a young woman about 19 or 20"),
    },
    "pitch": {
        "midlow": ("中等偏低", "音调中等偏低", "a slightly low-medium pitch"),
        "mid": ("中等", "音调中等", "a medium pitch"),
        "midhigh": ("中等偏高", "音调中等偏高", "a medium-high pitch"),
    },
    "texture": {
        "clear": ("清透", "声音干净清透", "a clean, clear voice"),
        "soft": ("柔和", "声音柔和，带一点点自然的气息感", "a soft voice with a touch of natural breath"),
        "warm": ("温润", "声音温润，有一点厚度", "a warm voice with a little body"),
        "grain": ("微颗粒", "声音带一点点沙沙的颗粒感", "a voice with a slight, subtle grain"),
    },
    "nasal": {
        "none": ("无", "", ""),
        "light": ("一点鼻音", "带一点点鼻音", "a slight nasal quality"),
    },
    "pace": {
        "natural": ("自然", "语速自然", "a natural speaking pace"),
        "relaxed": ("不紧不慢", "语速不紧不慢", "an unhurried pace"),
        "slightslow": ("稍慢", "语速稍慢一点", "a slightly slower pace"),
    },
    "mood": {
        "melancholy": ("淡淡心事", "语气里有一点淡淡的心事感", "a faint hint of melancholy in her tone"),
        "calm": ("平静", "语气平静温和", "a calm, gentle tone"),
        "tender": ("温柔", "语气温柔", "a tender, warm tone"),
        "distant": ("一点疏离", "语气安静，有一点点疏离", "a quiet tone, a little distant"),
        "sweetdown": ("微甜低落", "有一点甜，但情绪稍微有些低落", "a little sweet, but slightly down"),
    },
    "lazy": {
        "none": ("无", "", ""),
        "hint": ("细听才有", "细听才能察觉到一丝慵懒", "with a barely noticeable hint of laziness"),
        "light": ("轻微", "带一点轻微的慵懒和松弛感", "slightly lazy and relaxed"),
    },
    "manner": {
        "plain": ("不指定", "", ""),
        "chat": ("像聊天", "像在跟熟悉的朋友聊天", "like chatting with a close friend"),
        "smile": ("偶有笑意", "偶尔带一点笑意", "with an occasional hint of a smile"),
        "soft": ("轻轻说", "说话轻轻的", "speaking softly"),
    },
}

SLOT_ORDER = ["age", "pitch", "texture", "nasal", "pace", "mood", "lazy", "manner"]
SLOT_LABELS = {
    "age": "年龄感", "pitch": "音调", "texture": "质感", "nasal": "鼻音",
    "pace": "语速", "mood": "情绪底色", "lazy": "慵懒程度", "manner": "说话方式",
}

# 每条描述末尾都加上：把声音往「真人日常说话」拉，压一压人机感、播音腔和夹子音
DESC_SUFFIX = (
    "听起来就像真人在日常说话，口语自然，有自然的换气和语调起伏，不播音、不夹、不做作",
    "sounds like a real person talking in everyday life, conversational and natural, "
    "with natural breathing and intonation, not like a narrator, not exaggerated",
)

# 海选的 10 个方向（代号, 名字, 槽位）。A 是主推，其余围绕它探索。年龄以 17 岁为主，16、19-20 岁各留两个方向做对照。
ARCHETYPES = [
    ("A", "自然心事", dict(age="17", pitch="mid", texture="soft", nasal="none", pace="natural", mood="melancholy", lazy="hint", manner="chat")),
    ("B", "清透安静", dict(age="17", pitch="mid", texture="clear", nasal="none", pace="relaxed", mood="calm", lazy="hint", manner="plain")),
    ("C", "温柔低落", dict(age="16", pitch="midhigh", texture="soft", nasal="none", pace="relaxed", mood="sweetdown", lazy="hint", manner="soft")),
    ("D", "松弛日常", dict(age="17", pitch="mid", texture="warm", nasal="none", pace="natural", mood="tender", lazy="light", manner="chat")),
    ("E", "笑着藏心事", dict(age="17", pitch="midhigh", texture="clear", nasal="none", pace="natural", mood="melancholy", lazy="none", manner="smile")),
    ("F", "低温疏离", dict(age="19", pitch="midlow", texture="clear", nasal="none", pace="relaxed", mood="distant", lazy="hint", manner="plain")),
    ("G", "温润学姐", dict(age="19", pitch="mid", texture="warm", nasal="none", pace="relaxed", mood="tender", lazy="hint", manner="smile")),
    ("H", "颗粒独白", dict(age="17", pitch="midlow", texture="grain", nasal="none", pace="slightslow", mood="melancholy", lazy="hint", manner="soft")),
    ("I", "微甜鼻音", dict(age="16", pitch="mid", texture="soft", nasal="light", pace="natural", mood="sweetdown", lazy="hint", manner="chat")),
    ("J", "晚安电台", dict(age="17", pitch="mid", texture="warm", nasal="none", pace="slightslow", mood="calm", lazy="light", manner="soft")),
]

# 克隆测试用的台词：(标签, 语气控制, 台词)。语气控制只有 VoxCPM2 支持，留空则用高保真克隆。
# 数字写成汉字，避免读法不稳定。
CLONE_TEST_LINES = [
    ("日常", "", "今天的视频会有一点长，记得泡一杯热可可，慢慢看就好。"),
    ("忧郁", "语气有点低落，声音轻一些", "如果哪天我不见了，大概只是躲到哪个角落里，偷偷睡着了吧。"),
    ("可爱", "语气开心俏皮，带着笑意", "诶嘿，被你发现啦？那、那我把面具戴上好了！"),
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
    "夹/做作": ("texture", "清透"),
    "不够忧郁": ("mood", "淡淡心事"),
    "不够可爱": ("manner", "偶有笑意"),
    "哭腔重": ("mood", "平静"),
    "慵懒过重": ("lazy", "细听才有 / 无"),
    "人机感": (None, "这条描述本身可能没问题，换生成参数再试：VoxCPM2 试 --steps 20 或 --cfg 1.5"),
}
