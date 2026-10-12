#!/usr/bin/env python3
"""Write the V4 minimal-pair anchor draft (alternative-pairs/v4/pair_anchors.json).

Each pair: a populist claim and a matched mainstream sentence on the same topic,
with the same actors, grammatical person and roughly the same length, differing
in the construct-bearing proposition. For E the mainstream member is usually a
bounded, specific criticism of the same actor, so negative affect is matched
rather than contrasted. Drafted from theory; no benchmark label was consulted.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "pair_anchors.json"

P = [  # (subframe, topic, populist, matched_mainstream)
    ("popular_sovereignty", "referendum", "On this question the people must have the final word, not the politicians.", "On this question parliament will have the final vote, after a full debate."),
    ("popular_sovereignty", "election", "This election is about giving power back to the people who own this country.", "This election is about choosing the government that will lead this country."),
    ("popular_sovereignty", "courts", "No judge should be able to overrule what the people have clearly decided.", "No law should be passed without the courts being able to review it."),
    ("virtuous_common_people", "budget", "The people know what this country needs far better than any committee.", "The committee has studied what this country needs and published a report."),
    ("virtuous_common_people", "workers", "Ordinary working people are the backbone and the conscience of this nation.", "Working people in many sectors contribute to the economy of this nation."),
    ("popular_sovereignty", "european union", "Our people alone must decide our laws, not anyone in a distant capital.", "Our parliament will adopt these laws, in cooperation with our partners abroad."),
    ("popular_sovereignty", "pensions", "We will do what the people want on pensions, because they are the boss.", "We will reform pensions gradually, because they must remain balanced over time."),
    ("virtuous_common_people", "representation", "This movement is the true voice of the common people of this land.", "This party is one voice among many in the politics of this land."),
    ("popular_sovereignty", "government", "Every decision in this government will come from the will of the people.", "Every decision in this government will follow review by the relevant ministry."),
    ("popular_sovereignty", "democracy", "Real democracy means the people rule directly and their word is law.", "Real democracy means elected representatives decide and they answer to voters."),
    ("popular_sovereignty", "immigration", "On immigration the people have spoken and their will must be respected.", "On immigration the parliament has legislated and their rules are being applied."),
    ("virtuous_common_people", "families", "The decent hardworking families of this country deserve to be in charge.", "The many different families of this country deserve good public services."),
    ("popular_sovereignty", "constitution", "Sovereignty belongs to the people and no institution can take it from them.", "Sovereignty is exercised through institutions, and the constitution establishes them."),
    ("popular_sovereignty", "energy", "The people should decide our energy future, not unelected experts.", "The energy agency should advise on our future, and parliament decides."),
    ("popular_sovereignty", "mobilization", "We are the people and together we will take this country back.", "We are a coalition and together we will govern this country responsibly."),
    ("virtuous_common_people", "health", "Our people built this health service and they must be the ones who run it.", "Our staff run this health service and they report to the board that oversees it."),
    ("popular_sovereignty", "treaty", "Any treaty must be put to the people, because only they can approve it.", "Any treaty must be ratified by parliament, because only they can approve it."),
    ("virtuous_common_people", "expertise", "The common sense of ordinary people is worth more than all the reports.", "The analysis in these reports will inform the decision taken next month."),
    ("popular_sovereignty", "mandate", "The people gave us a mandate and nothing can stand in the way of it.", "The voters gave us a majority and we will work with the other parties."),
    ("popular_sovereignty", "decentralisation", "Power must be taken from the offices and handed back to the people.", "Powers will be shared between the national offices and local councils."),
    ("virtuous_common_people", "unity", "There is one people with one will, and we are here to carry it out.", "There are many views in this country, and we are here to balance those views."),
    ("popular_sovereignty", "taxes", "It is the people's money and the people alone should say how it is spent.", "It is public money and parliament should scrutinise how it is spent."),
    ("virtuous_common_people", "heartland", "The real people of this country live in its towns, farms and factories.", "The people of this country live in its cities, towns and rural areas."),
    ("popular_sovereignty", "referendum result", "When the people vote, that is the end of the argument for everyone.", "When the people vote, the result is implemented according to the law."),
    ("popular_sovereignty", "governing style", "I will govern by listening to the people and to nobody else.", "I will govern by consulting experts, parties and the public."),
    ("popular_sovereignty", "schools", "Parents, not bureaucrats, must decide what our children learn.", "Parents and teachers, with the ministry, shape what our children learn."),
    ("virtuous_common_people", "justice", "Justice must serve the ordinary people first and always.", "Justice must be applied equally and impartially to all."),
    ("popular_sovereignty", "nation", "The nation belongs to its people and their will is sacred.", "The nation is governed by its laws and their rule is binding."),
    ("virtuous_common_people", "direction", "The workers of this country are the ones who must decide its direction.", "The government of this country will set out its direction in a programme."),
    ("popular_sovereignty", "protest", "When the people fill the streets, the government must simply obey them.", "When citizens protest peacefully, the government should listen to them."),
    ("popular_sovereignty", "consultation", "We do not need more consultations, we need to do what the people say.", "We will hold further consultations before deciding what we should do."),
    ("virtuous_common_people", "wisdom", "Simple people have more wisdom than all the professors in the capital.", "Many people have useful views, and the professors add expertise to the debate."),
    ("popular_sovereignty", "majority", "The majority is the people and the people's will must always prevail.", "The majority governs but minority rights must always be protected."),
    ("popular_sovereignty", "authority", "No authority on earth stands above the free people of this nation.", "Every authority in this nation is bound by the constitution and courts."),
    ("virtuous_common_people", "future", "The future of this country will be written by its ordinary citizens.", "The future of this country will be shaped by many policy choices."),
    ("virtuous_common_people", "ownership", "Our farmers and workers are the true owners of this republic.", "Our farmers and workers are important sectors of this economy."),
    ("popular_sovereignty", "decision", "Let the people decide, and whatever they choose, that is final.", "Let the committee report, and whatever they recommend, the chamber decides."),
    ("popular_sovereignty", "accountability", "My only master is the people and I answer to no one else.", "My ministry answers to parliament and to the auditor general."),
    ("popular_sovereignty", "restoration", "We will restore government of the people, by the people, for the people.", "We will restore good government that is efficient, fair and accountable."),
    ("virtuous_common_people", "heart of the country", "Ordinary men and women are the heart and soul of this country.", "Businesses, charities and public bodies all contribute to this country."),
]

E = [
    ("institutional_capture", "courts", "A corrupt clique has captured the courts and uses them against ordinary people.", "A backlog of cases has slowed the courts, and they now serve ordinary people late."),
    ("institutional_capture", "elections", "The system is rigged by insiders so that ordinary voters can never win.", "The system has flaws in districting so that some votes carry less weight."),
    ("institutional_capture", "bureaucracy", "Unelected bureaucrats run this country and ignore everything the public wants.", "Civil servants administer this programme and report on it to parliament each year."),
    ("corruption_self_enrichment", "contracts", "They hand every contract to their cronies and you pay the bill.", "They awarded the contract after a review, and you will see the costs published."),
    ("corruption_self_enrichment", "salaries", "The politicians fill their pockets while the people are told to tighten their belts.", "The politicians approved their pay rise, and the public debate about it continues."),
    ("betrayal_of_the_nation", "factory", "The elites in the capital sold out our workers to line their own pockets.", "The government in the capital did not protect our workers, and their plant closed."),
    ("closed_political_caste", "political class", "The political class is a closed caste that lives by its own rules.", "The political parties are recruiting candidates from different backgrounds."),
    ("closed_political_caste", "voting", "It makes no difference who you vote for, the same elite always wins.", "It makes a difference who you vote for, the parties offer real choices."),
    ("media_establishment", "press", "The establishment media lie to protect the powerful and deceive the people.", "The newspaper made an error in its report and has since corrected it."),
    ("media_establishment", "broadcasters", "The big broadcasters are mouthpieces of the establishment against the public.", "The public broadcaster is independent and its coverage is regularly reviewed."),
    ("supranational_technocracy", "brussels", "Faceless technocrats in Brussels dictate to our people and answer to no one.", "The commission in Brussels proposed new rules and our ministers will negotiate."),
    ("financial_economic_elite", "imf", "The bankers and the IMF have enslaved our people with their debt.", "Our loan from the IMF comes with their conditions, which will be difficult."),
    ("financial_economic_elite", "wall street", "Wall Street owns Washington and the system works only for the rich.", "Wall Street is regulated by Washington and the rules need updating."),
    ("institutional_capture", "lobbying", "Lobbyists and donors have bought the government and silenced the voters.", "Lobbyists and donors must register, and the disclosure rules are now tighter."),
    ("financial_economic_elite", "bailout", "The banks were rescued while ordinary families were thrown out of their homes.", "The banks were recapitalised and a scheme helped families keep their homes."),
    ("institutional_capture", "regulators", "The regulators are in the pocket of the corporations they are meant to police.", "The regulators missed warning signs at the corporations they are meant to supervise."),
    ("financial_economic_elite", "oligarchs", "A handful of oligarchs own this country and treat the people as servants.", "A handful of firms dominate this market and the competition law will apply."),
    ("closed_political_caste", "cartel parties", "The old parties are one cartel that shares power and robs the nation.", "The old parties have formed a coalition and will share the ministries."),
    ("institutional_capture", "experts", "The so-called experts serve the powerful and despise ordinary people.", "The advisory experts reached a conclusion that many people dispute."),
    ("corruption_self_enrichment", "energy prices", "The energy giants and their friends in government are fleecing you.", "The energy companies raised your tariffs and the regulator is reviewing their pricing."),
    ("closed_political_caste", "privileges", "The ruling elite protects its privileges and cares nothing for the people.", "The ruling coalition has kept its promises on some issues but not others."),
    ("institutional_capture", "secrecy", "Behind closed doors a small circle decides everything against your interests.", "In committee meetings a small group prepares the bills that affect your interests."),
    ("betrayal_of_the_nation", "neglect", "The people in the capital have forgotten you and work only for themselves.", "The ministries in the capital have delayed your regional projects and they must explain why."),
    ("corruption_self_enrichment", "corruption", "Corruption is not an accident here, it is how the elite rules the people.", "Corruption cases here are prosecuted, and the new law adds tougher penalties."),
    ("financial_economic_elite", "globalisation", "Global elites are tearing down our nation for their own profit.", "Global markets are changing our industries and their effects are uneven."),
    ("institutional_capture", "judges", "Activist judges and their friends overturn the will of the voters.", "The judges reviewed the law and their ruling found parts unconstitutional."),
    ("institutional_capture", "reform", "The deep state blocks every reform that would help ordinary families.", "The administration delayed the reform, and the timetable has slipped."),
    ("corruption_self_enrichment", "pension fund", "The elite raided your pension fund to pay for their own mistakes.", "The government used your pension reserve to cover their budget deficit."),
    ("closed_political_caste", "party bosses", "The party bosses decide everything and treat the members like fools.", "The party executive sets the agenda and the members vote on motions."),
    ("corruption_self_enrichment", "health bonuses", "The health bosses pay themselves bonuses while patients die in corridors.", "The health board paid their managers bonuses last year while waiting times grew."),
    ("media_establishment", "media owners", "A few billionaires own the press and tell you what to think.", "A few companies own most of the newspapers you read, and the market is concentrated."),
    ("betrayal_of_the_nation", "betrayal", "The people in power have betrayed everyone who ever trusted them.", "The people in power have missed several targets they set themselves."),
    ("closed_political_caste", "contempt", "They look down on you and think you are too stupid to decide.", "They disagree with you and think the policy needs more work."),
    ("institutional_capture", "institutions", "Every institution has been taken over by the same corrupt network.", "Several institutions have been reorganised under the new legal framework."),
    ("financial_economic_elite", "tax", "The rich and powerful write the tax laws so that only you pay.", "The tax code has loopholes, so you pay more than some large firms."),
    ("supranational_technocracy", "eu budget", "Brussels bureaucrats take our money and use it against our nation.", "The European budget includes our contribution and funds our regions."),
    ("corruption_self_enrichment", "revolving door", "Ministers retire into the boardrooms that paid for their favours.", "Ministers who retire must wait two years before they join companies."),
    ("institutional_capture", "campaign finance", "Big money has captured our democracy and the voters are for sale.", "Campaign spending in our elections has risen and the limits should be reviewed."),
    ("corruption_self_enrichment", "public office", "The officials treat public office as their own private property.", "The officials manage public office according to their service code."),
    ("betrayal_of_the_nation", "crisis", "They created this crisis to enrich themselves at your expense.", "They mishandled this crisis, and the inquiry will tell you why."),
]

M = [
    ("moral_condemnation", "framing", "This is a battle between good and evil, and they are on the side of evil.", "This is a debate between two approaches, and they favour the other one."),
    ("deception_and_lies", "opponents", "Our opponents are liars and cheats who have no shame at all.", "Our opponents are mistaken on this point and we will say why."),
    ("moral_condemnation", "treaty", "They are traitors to this nation and history will judge them harshly.", "They are wrong on this treaty and the vote will decide the matter."),
    ("moral_condemnation", "compromise", "There can be no compromise with people who are this dishonest.", "There can be a compromise with people who hold this view."),
    ("moral_condemnation", "character", "They are rotten to the core and everything they touch is poisoned.", "They are new to this policy and some of their ideas are untested."),
    ("deception_and_lies", "trust", "Their lies have destroyed trust and they must be driven from office.", "Their figures are disputed and they should publish the full data."),
    ("moral_condemnation", "us and them", "We are the honest ones and they are the parasites feeding on us.", "We have one plan and they have another that also has supporters."),
    ("moral_condemnation", "intent", "These people are not mistaken, they are wicked.", "These people are not our allies, but they are sincere."),
    ("deception_and_lies", "claims", "Every word they say is a deliberate deception.", "Several claims they make are open to question."),
    ("moral_condemnation", "morals", "They have no morals and would sell their own mothers for power.", "They have other priorities and would spend the money elsewhere."),
    ("moral_condemnation", "struggle", "This is a struggle of decent citizens against a criminal gang.", "This is a disagreement between colleagues about the right priorities."),
    ("moral_condemnation", "ruin", "Their greed and their treachery have brought this country to ruin.", "Their policy and their timing have brought this country some problems."),
    ("moral_condemnation", "enemies", "They are enemies of the people and they must be stopped.", "They are members of the opposition and they will be answered."),
    ("moral_condemnation", "plan", "Only villains could defend such a monstrous plan.", "Only a few members could support such an expensive plan."),
    ("moral_condemnation", "right and wrong", "The other side is evil and we are fighting for what is right.", "The other side is honest and we simply see the issue differently."),
    ("deception_and_lies", "hypocrisy", "They are hypocrites who preach virtue and practise vice.", "They are colleagues who argue their case and vote their conscience."),
    ("moral_condemnation", "nature", "Their corruption is not an accident, it is who they are.", "Their proposal is not a surprise, it is what they promised."),
    ("moral_condemnation", "cooperation", "We will never shake hands with these scoundrels.", "We will always work with the members opposite."),
    ("moral_condemnation", "deals", "Shame on them for their cowardice and their filthy deals.", "Credit to them for their effort, though we disagree on the deal."),
    ("moral_condemnation", "dishonour", "Their hands are stained with dishonour and they know it.", "Their plan is based on assumptions and they know that too."),
    ("deception_and_lies", "politics as war", "This is not politics, it is a war between truth and lies.", "This is ordinary politics, a contest between different programmes."),
    ("deception_and_lies", "decades", "Those crooks have been lying to your face for decades.", "Those ministers have been working on this issue for years."),
    ("moral_condemnation", "fitness", "They are morally bankrupt and unfit to lead anyone.", "They are experienced ministers and able to lead a department."),
    ("deception_and_lies", "honesty", "Nothing they do is honest, everything is a trick.", "Some of what they do is sensible, some needs change."),
    ("deception_and_lies", "reform", "Their so-called reform is theft dressed up as progress.", "Their proposed reform is costly but it has some merits."),
    ("moral_condemnation", "majority", "The decent majority is being held hostage by a gang of frauds.", "The governing majority is being challenged by a strong opposition."),
    ("moral_condemnation", "history", "They spit on everything our grandparents fought for.", "They see our history differently from how we see it."),
    ("moral_condemnation", "adversaries", "Those who oppose us are not opponents, they are enemies.", "Those who oppose us are not enemies, they are opponents."),
    ("moral_condemnation", "families", "They are vultures circling over the suffering of families.", "They are critics raising concerns about support for families."),
    ("deception_and_lies", "conscience", "Any person of conscience must reject these liars completely.", "Any person of good will can accept parts of this proposal."),
    ("deception_and_lies", "campaign", "Their dirty tricks show how low and corrupt they really are.", "Their campaign tactics show how hard they are competing for votes."),
    ("moral_condemnation", "responsibility", "They have blood on their hands and no apology can wash it off.", "They made a mistake on this and they have apologised for it."),
    ("moral_condemnation", "sin", "It is a sin what they have done to the people of this land.", "It is a pity what happened in the negotiations this week."),
    ("moral_condemnation", "purity", "We are pure and they are poison, and the voters know it.", "We are ready and they are ready, and the voters will choose."),
    ("moral_condemnation", "agenda", "Their evil agenda must be defeated once and for all.", "Their policy agenda must be debated and put to a vote."),
    ("deception_and_lies", "sincerity", "They are cynical frauds pretending to care about you.", "They are serious people who care about this issue too."),
    ("deception_and_lies", "credibility", "No decent person could trust a single word from these cheats.", "Most reasonable people would weigh the arguments from both sides."),
    ("deception_and_lies", "hatred", "They have poisoned our politics with lies and hatred.", "They have changed our politics with new ideas and methods."),
    ("moral_condemnation", "choice", "Honest citizens against a rotten establishment, that is the choice.", "Two credible parties with different programmes, that is the choice."),
    ("deception_and_lies", "public life", "They are the most dishonest people I have ever met in public life.", "They are the most determined people I have ever met in public life."),
]

LABELS = {"P": "people sovereignty", "E": "elite obstruction", "M": "moralized antagonism"}


def main() -> None:
    channels = {}
    for channel, rows in (("P", P), ("E", E), ("M", M)):
        channels[channel] = {
            "label": LABELS[channel],
            "pairs": [
                {
                    "pair_id": f"{channel}{index:02d}",
                    "subframe": subframe,
                    "topic": topic,
                    "populist": populist,
                    "matched_mainstream": mainstream,
                }
                for index, (subframe, topic, populist, mainstream) in enumerate(rows, start=1)
            ],
        }
    payload = {
        "schema": "populism_axis.v4.pair_anchors.v1",
        "anchor_set_id": "pa_v4_pair_anchors_r1",
        "status": "reviewed",
        "outcome_labels_used": False,
        "construct_source": "Ideational definition of populism: people-centrism (P), anti-elitism (E), Manichean moral antagonism (M). Sub-frame tags follow alternative-pairs/v3/subframe_anchors.json.",
        "drafting_note": (
            "Minimal pairs: each mainstream member keeps the populist member's topic, actors, grammatical person "
            "and approximate length, and replaces only the construct-bearing claim. For E the mainstream member is "
            "usually a bounded, specific criticism of the same actor, so negative affect is matched rather than "
            "contrasted. M contrasts moralized antagonism with ordinary disagreement and is expected to differ in "
            "sentiment by construction. Drafted from theory before any step-3 result; no benchmark label was consulted."
        ),
        "review": {'reviewer': 'Aleksandar Tomasevic (project lead)', 'date': '2026-09-23', 'outcome': 'approved', 'changes': 'none'},
        "channels": channels,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: " + ", ".join(f"{c}={len(v['pairs'])}" for c, v in channels.items()))


if __name__ == "__main__":
    main()
