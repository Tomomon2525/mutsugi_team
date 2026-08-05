"""自動生成。直接編集しない。tools/gen_enums.py で再生成する。

公式エンジンのヘッダから抽出した機構名の定数である。カード名・カードテキストは含まない。
"""

from enum import IntEnum


class SelectType(IntEnum):
    Main = 0
    Card = 1
    AttachedCard = 2
    CardOrAttachedCard = 3
    Energy = 4
    Skill = 5
    Attack = 6
    Evolve = 7
    Count = 8
    YesNo = 9
    SpecialCondition = 10


class SelectContext(IntEnum):
    Main = 0
    SetupActivePokemon = 1
    SetupBenchPokemon = 2
    Switch = 3
    ToActive = 4
    ToBench = 5
    ToField = 6
    ToHand = 7
    Discard = 8
    ToDeck = 9
    ToDeckBottom = 10
    ToPrize = 11
    NotMove = 12
    DamageCounter = 13
    DamageCounterAny = 14
    Damage = 15
    RemoveDamageCounter = 16
    Heal = 17
    EvolvesFrom = 18
    EvolvesTo = 19
    Devolve = 20
    AttachFrom = 21
    AttachTo = 22
    DetachFrom = 23
    Look = 24
    EffectTarget = 25
    DiscardEnergyCard = 26
    DiscardToolCard = 27
    SwitchEnergyCard = 28
    DiscardCardOrAttachedCard = 29
    DiscardEnergy = 30
    ToHandEnergy = 31
    ToDeckEnergy = 32
    SwitchEnergy = 33
    SkillOrder = 34
    Attack = 35
    DisableAttack = 36
    Evolve = 37
    DrawCount = 38
    DamageCounterCount = 39
    RemoveDamageCounterCount = 40
    IsFirst = 41
    Mulligan = 42
    Activate = 43
    FirstEffect = 44
    MoreDevolve = 45
    CoinHead = 46
    AffectSpecialCondition = 47
    RecoverSpecialCondition = 48


class OptionType(IntEnum):
    Number = 0
    Yes = 1
    No = 2
    Card = 3
    ToolCard = 4
    EnergyCard = 5
    Energy = 6
    Play = 7
    Attach = 8
    Evolve = 9
    Ability = 10
    Discard = 11
    Retreat = 12
    Attack = 13
    End = 14
    Skill = 15
    SpecialCondition = 16


class LogType(IntEnum):
    Shuffle = 0
    HasBasicPokemon = 1
    TurnStart = 2
    TurnEnd = 3
    Draw = 4
    DrawReverse = 5
    MoveCard = 6
    MoveCardReverse = 7
    Switch = 8
    Change = 9
    Play = 10
    Attach = 11
    Evolve = 12
    Devolve = 13
    MoveAttached = 14
    Attack = 15
    HpChange = 16
    Poisoned = 17
    Burned = 18
    Asleep = 19
    Paralyzed = 20
    Confused = 21
    Coin = 22
    Result = 23


class AreaType(IntEnum):
    All = 0
    Deck = 1
    Hand = 2
    Trash = 3
    Active = 4
    Bench = 5
    Prize = 6
    Stadium = 7
    Energy = 8
    Tool = 9
    PreEvolution = 10
    Player = 11
    Looking = 12
    Playing = 13
    DeckBottom = 14
    Me = 15
    Effected = 16
    EffectedPreTarget = 17
    SelectedList = 18
    TriggerSubject = 19
    TriggerObject = 20
    Attach = 21
    TurnPlay = 22
    AttackPreMyTurn = 23
    Temporary = 24


class CardType(IntEnum):
    Pokemon = 0
    Item = 1
    Tool = 2
    Supporter = 3
    Stadium = 4
    BasicEnergy = 5
    SpecialEnergy = 6


class PokemonType(IntEnum):
    NotPokemon = 0
    Normal = 1
    PokemonItem = 2
    Ex = 3
    MegaEx = 4


class EvolutionType(IntEnum):
    NoEvolutionType = 0
    Basic = 1
    Stage1 = 2
    Stage2 = 3


class EnergyIndex(IntEnum):
    Colorless = 0
    Grass = 1
    Fire = 2
    Water = 3
    Lightning = 4
    Psychic = 5
    Fighting = 6
    Darkness = 7
    Metal = 8
    Dragon = 9
