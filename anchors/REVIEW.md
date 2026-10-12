# V4 anchor and nuisance drafts: review sheet

Status: **reviewed and approved by the project lead on 2026-09-23, no changes**. Edit the source files, not this sheet:

- minimal pairs: `alternative-pairs/v4/build_pair_anchors_draft.py` (writes `pair_anchors.json`)
- nuisance sets: `alternative-pairs/v4/nuisance_sentences.json`
- balance report: `alternative-pairs/v4/anchor_balance_report.md`

What to check:

1. Each populist sentence states the channel's claim and nothing else from another channel where avoidable.
2. Each matched mainstream sentence is the same topic and actors with only the populist claim replaced.
3. No nuisance sentence contains a people-versus-elite claim (otherwise residualization removes real signal).
4. Coverage: topics and sub-frames you consider missing, and left/right balance of targets.

## Channel P: people sovereignty (40 pairs)

| ID | Sub-frame | Topic | Populist | Matched mainstream |
|---|---|---|---|---|
| P01 | popular_sovereignty | referendum | On this question the people must have the final word, not the politicians. | On this question parliament will have the final vote, after a full debate. |
| P02 | popular_sovereignty | election | This election is about giving power back to the people who own this country. | This election is about choosing the government that will lead this country. |
| P03 | popular_sovereignty | courts | No judge should be able to overrule what the people have clearly decided. | No law should be passed without the courts being able to review it. |
| P04 | virtuous_common_people | budget | The people know what this country needs far better than any committee. | The committee has studied what this country needs and published a report. |
| P05 | virtuous_common_people | workers | Ordinary working people are the backbone and the conscience of this nation. | Working people in many sectors contribute to the economy of this nation. |
| P06 | popular_sovereignty | european union | Our people alone must decide our laws, not anyone in a distant capital. | Our parliament will adopt these laws, in cooperation with our partners abroad. |
| P07 | popular_sovereignty | pensions | We will do what the people want on pensions, because they are the boss. | We will reform pensions gradually, because they must remain balanced over time. |
| P08 | virtuous_common_people | representation | This movement is the true voice of the common people of this land. | This party is one voice among many in the politics of this land. |
| P09 | popular_sovereignty | government | Every decision in this government will come from the will of the people. | Every decision in this government will follow review by the relevant ministry. |
| P10 | popular_sovereignty | democracy | Real democracy means the people rule directly and their word is law. | Real democracy means elected representatives decide and they answer to voters. |
| P11 | popular_sovereignty | immigration | On immigration the people have spoken and their will must be respected. | On immigration the parliament has legislated and their rules are being applied. |
| P12 | virtuous_common_people | families | The decent hardworking families of this country deserve to be in charge. | The many different families of this country deserve good public services. |
| P13 | popular_sovereignty | constitution | Sovereignty belongs to the people and no institution can take it from them. | Sovereignty is exercised through institutions, and the constitution establishes them. |
| P14 | popular_sovereignty | energy | The people should decide our energy future, not unelected experts. | The energy agency should advise on our future, and parliament decides. |
| P15 | popular_sovereignty | mobilization | We are the people and together we will take this country back. | We are a coalition and together we will govern this country responsibly. |
| P16 | virtuous_common_people | health | Our people built this health service and they must be the ones who run it. | Our staff run this health service and they report to the board that oversees it. |
| P17 | popular_sovereignty | treaty | Any treaty must be put to the people, because only they can approve it. | Any treaty must be ratified by parliament, because only they can approve it. |
| P18 | virtuous_common_people | expertise | The common sense of ordinary people is worth more than all the reports. | The analysis in these reports will inform the decision taken next month. |
| P19 | popular_sovereignty | mandate | The people gave us a mandate and nothing can stand in the way of it. | The voters gave us a majority and we will work with the other parties. |
| P20 | popular_sovereignty | decentralisation | Power must be taken from the offices and handed back to the people. | Powers will be shared between the national offices and local councils. |
| P21 | virtuous_common_people | unity | There is one people with one will, and we are here to carry it out. | There are many views in this country, and we are here to balance those views. |
| P22 | popular_sovereignty | taxes | It is the people's money and the people alone should say how it is spent. | It is public money and parliament should scrutinise how it is spent. |
| P23 | virtuous_common_people | heartland | The real people of this country live in its towns, farms and factories. | The people of this country live in its cities, towns and rural areas. |
| P24 | popular_sovereignty | referendum result | When the people vote, that is the end of the argument for everyone. | When the people vote, the result is implemented according to the law. |
| P25 | popular_sovereignty | governing style | I will govern by listening to the people and to nobody else. | I will govern by consulting experts, parties and the public. |
| P26 | popular_sovereignty | schools | Parents, not bureaucrats, must decide what our children learn. | Parents and teachers, with the ministry, shape what our children learn. |
| P27 | virtuous_common_people | justice | Justice must serve the ordinary people first and always. | Justice must be applied equally and impartially to all. |
| P28 | popular_sovereignty | nation | The nation belongs to its people and their will is sacred. | The nation is governed by its laws and their rule is binding. |
| P29 | virtuous_common_people | direction | The workers of this country are the ones who must decide its direction. | The government of this country will set out its direction in a programme. |
| P30 | popular_sovereignty | protest | When the people fill the streets, the government must simply obey them. | When citizens protest peacefully, the government should listen to them. |
| P31 | popular_sovereignty | consultation | We do not need more consultations, we need to do what the people say. | We will hold further consultations before deciding what we should do. |
| P32 | virtuous_common_people | wisdom | Simple people have more wisdom than all the professors in the capital. | Many people have useful views, and the professors add expertise to the debate. |
| P33 | popular_sovereignty | majority | The majority is the people and the people's will must always prevail. | The majority governs but minority rights must always be protected. |
| P34 | popular_sovereignty | authority | No authority on earth stands above the free people of this nation. | Every authority in this nation is bound by the constitution and courts. |
| P35 | virtuous_common_people | future | The future of this country will be written by its ordinary citizens. | The future of this country will be shaped by many policy choices. |
| P36 | virtuous_common_people | ownership | Our farmers and workers are the true owners of this republic. | Our farmers and workers are important sectors of this economy. |
| P37 | popular_sovereignty | decision | Let the people decide, and whatever they choose, that is final. | Let the committee report, and whatever they recommend, the chamber decides. |
| P38 | popular_sovereignty | accountability | My only master is the people and I answer to no one else. | My ministry answers to parliament and to the auditor general. |
| P39 | popular_sovereignty | restoration | We will restore government of the people, by the people, for the people. | We will restore good government that is efficient, fair and accountable. |
| P40 | virtuous_common_people | heart of the country | Ordinary men and women are the heart and soul of this country. | Businesses, charities and public bodies all contribute to this country. |

## Channel E: elite obstruction (40 pairs)

| ID | Sub-frame | Topic | Populist | Matched mainstream |
|---|---|---|---|---|
| E01 | institutional_capture | courts | A corrupt clique has captured the courts and uses them against ordinary people. | A backlog of cases has slowed the courts, and they now serve ordinary people late. |
| E02 | institutional_capture | elections | The system is rigged by insiders so that ordinary voters can never win. | The system has flaws in districting so that some votes carry less weight. |
| E03 | institutional_capture | bureaucracy | Unelected bureaucrats run this country and ignore everything the public wants. | Civil servants administer this programme and report on it to parliament each year. |
| E04 | corruption_self_enrichment | contracts | They hand every contract to their cronies and you pay the bill. | They awarded the contract after a review, and you will see the costs published. |
| E05 | corruption_self_enrichment | salaries | The politicians fill their pockets while the people are told to tighten their belts. | The politicians approved their pay rise, and the public debate about it continues. |
| E06 | betrayal_of_the_nation | factory | The elites in the capital sold out our workers to line their own pockets. | The government in the capital did not protect our workers, and their plant closed. |
| E07 | closed_political_caste | political class | The political class is a closed caste that lives by its own rules. | The political parties are recruiting candidates from different backgrounds. |
| E08 | closed_political_caste | voting | It makes no difference who you vote for, the same elite always wins. | It makes a difference who you vote for, the parties offer real choices. |
| E09 | media_establishment | press | The establishment media lie to protect the powerful and deceive the people. | The newspaper made an error in its report and has since corrected it. |
| E10 | media_establishment | broadcasters | The big broadcasters are mouthpieces of the establishment against the public. | The public broadcaster is independent and its coverage is regularly reviewed. |
| E11 | supranational_technocracy | brussels | Faceless technocrats in Brussels dictate to our people and answer to no one. | The commission in Brussels proposed new rules and our ministers will negotiate. |
| E12 | financial_economic_elite | imf | The bankers and the IMF have enslaved our people with their debt. | Our loan from the IMF comes with their conditions, which will be difficult. |
| E13 | financial_economic_elite | wall street | Wall Street owns Washington and the system works only for the rich. | Wall Street is regulated by Washington and the rules need updating. |
| E14 | institutional_capture | lobbying | Lobbyists and donors have bought the government and silenced the voters. | Lobbyists and donors must register, and the disclosure rules are now tighter. |
| E15 | financial_economic_elite | bailout | The banks were rescued while ordinary families were thrown out of their homes. | The banks were recapitalised and a scheme helped families keep their homes. |
| E16 | institutional_capture | regulators | The regulators are in the pocket of the corporations they are meant to police. | The regulators missed warning signs at the corporations they are meant to supervise. |
| E17 | financial_economic_elite | oligarchs | A handful of oligarchs own this country and treat the people as servants. | A handful of firms dominate this market and the competition law will apply. |
| E18 | closed_political_caste | cartel parties | The old parties are one cartel that shares power and robs the nation. | The old parties have formed a coalition and will share the ministries. |
| E19 | institutional_capture | experts | The so-called experts serve the powerful and despise ordinary people. | The advisory experts reached a conclusion that many people dispute. |
| E20 | corruption_self_enrichment | energy prices | The energy giants and their friends in government are fleecing you. | The energy companies raised your tariffs and the regulator is reviewing their pricing. |
| E21 | closed_political_caste | privileges | The ruling elite protects its privileges and cares nothing for the people. | The ruling coalition has kept its promises on some issues but not others. |
| E22 | institutional_capture | secrecy | Behind closed doors a small circle decides everything against your interests. | In committee meetings a small group prepares the bills that affect your interests. |
| E23 | betrayal_of_the_nation | neglect | The people in the capital have forgotten you and work only for themselves. | The ministries in the capital have delayed your regional projects and they must explain why. |
| E24 | corruption_self_enrichment | corruption | Corruption is not an accident here, it is how the elite rules the people. | Corruption cases here are prosecuted, and the new law adds tougher penalties. |
| E25 | financial_economic_elite | globalisation | Global elites are tearing down our nation for their own profit. | Global markets are changing our industries and their effects are uneven. |
| E26 | institutional_capture | judges | Activist judges and their friends overturn the will of the voters. | The judges reviewed the law and their ruling found parts unconstitutional. |
| E27 | institutional_capture | reform | The deep state blocks every reform that would help ordinary families. | The administration delayed the reform, and the timetable has slipped. |
| E28 | corruption_self_enrichment | pension fund | The elite raided your pension fund to pay for their own mistakes. | The government used your pension reserve to cover their budget deficit. |
| E29 | closed_political_caste | party bosses | The party bosses decide everything and treat the members like fools. | The party executive sets the agenda and the members vote on motions. |
| E30 | corruption_self_enrichment | health bonuses | The health bosses pay themselves bonuses while patients die in corridors. | The health board paid their managers bonuses last year while waiting times grew. |
| E31 | media_establishment | media owners | A few billionaires own the press and tell you what to think. | A few companies own most of the newspapers you read, and the market is concentrated. |
| E32 | betrayal_of_the_nation | betrayal | The people in power have betrayed everyone who ever trusted them. | The people in power have missed several targets they set themselves. |
| E33 | closed_political_caste | contempt | They look down on you and think you are too stupid to decide. | They disagree with you and think the policy needs more work. |
| E34 | institutional_capture | institutions | Every institution has been taken over by the same corrupt network. | Several institutions have been reorganised under the new legal framework. |
| E35 | financial_economic_elite | tax | The rich and powerful write the tax laws so that only you pay. | The tax code has loopholes, so you pay more than some large firms. |
| E36 | supranational_technocracy | eu budget | Brussels bureaucrats take our money and use it against our nation. | The European budget includes our contribution and funds our regions. |
| E37 | corruption_self_enrichment | revolving door | Ministers retire into the boardrooms that paid for their favours. | Ministers who retire must wait two years before they join companies. |
| E38 | institutional_capture | campaign finance | Big money has captured our democracy and the voters are for sale. | Campaign spending in our elections has risen and the limits should be reviewed. |
| E39 | corruption_self_enrichment | public office | The officials treat public office as their own private property. | The officials manage public office according to their service code. |
| E40 | betrayal_of_the_nation | crisis | They created this crisis to enrich themselves at your expense. | They mishandled this crisis, and the inquiry will tell you why. |

## Channel M: moralized antagonism (40 pairs)

| ID | Sub-frame | Topic | Populist | Matched mainstream |
|---|---|---|---|---|
| M01 | moral_condemnation | framing | This is a battle between good and evil, and they are on the side of evil. | This is a debate between two approaches, and they favour the other one. |
| M02 | deception_and_lies | opponents | Our opponents are liars and cheats who have no shame at all. | Our opponents are mistaken on this point and we will say why. |
| M03 | moral_condemnation | treaty | They are traitors to this nation and history will judge them harshly. | They are wrong on this treaty and the vote will decide the matter. |
| M04 | moral_condemnation | compromise | There can be no compromise with people who are this dishonest. | There can be a compromise with people who hold this view. |
| M05 | moral_condemnation | character | They are rotten to the core and everything they touch is poisoned. | They are new to this policy and some of their ideas are untested. |
| M06 | deception_and_lies | trust | Their lies have destroyed trust and they must be driven from office. | Their figures are disputed and they should publish the full data. |
| M07 | moral_condemnation | us and them | We are the honest ones and they are the parasites feeding on us. | We have one plan and they have another that also has supporters. |
| M08 | moral_condemnation | intent | These people are not mistaken, they are wicked. | These people are not our allies, but they are sincere. |
| M09 | deception_and_lies | claims | Every word they say is a deliberate deception. | Several claims they make are open to question. |
| M10 | moral_condemnation | morals | They have no morals and would sell their own mothers for power. | They have other priorities and would spend the money elsewhere. |
| M11 | moral_condemnation | struggle | This is a struggle of decent citizens against a criminal gang. | This is a disagreement between colleagues about the right priorities. |
| M12 | moral_condemnation | ruin | Their greed and their treachery have brought this country to ruin. | Their policy and their timing have brought this country some problems. |
| M13 | moral_condemnation | enemies | They are enemies of the people and they must be stopped. | They are members of the opposition and they will be answered. |
| M14 | moral_condemnation | plan | Only villains could defend such a monstrous plan. | Only a few members could support such an expensive plan. |
| M15 | moral_condemnation | right and wrong | The other side is evil and we are fighting for what is right. | The other side is honest and we simply see the issue differently. |
| M16 | deception_and_lies | hypocrisy | They are hypocrites who preach virtue and practise vice. | They are colleagues who argue their case and vote their conscience. |
| M17 | moral_condemnation | nature | Their corruption is not an accident, it is who they are. | Their proposal is not a surprise, it is what they promised. |
| M18 | moral_condemnation | cooperation | We will never shake hands with these scoundrels. | We will always work with the members opposite. |
| M19 | moral_condemnation | deals | Shame on them for their cowardice and their filthy deals. | Credit to them for their effort, though we disagree on the deal. |
| M20 | moral_condemnation | dishonour | Their hands are stained with dishonour and they know it. | Their plan is based on assumptions and they know that too. |
| M21 | deception_and_lies | politics as war | This is not politics, it is a war between truth and lies. | This is ordinary politics, a contest between different programmes. |
| M22 | deception_and_lies | decades | Those crooks have been lying to your face for decades. | Those ministers have been working on this issue for years. |
| M23 | moral_condemnation | fitness | They are morally bankrupt and unfit to lead anyone. | They are experienced ministers and able to lead a department. |
| M24 | deception_and_lies | honesty | Nothing they do is honest, everything is a trick. | Some of what they do is sensible, some needs change. |
| M25 | deception_and_lies | reform | Their so-called reform is theft dressed up as progress. | Their proposed reform is costly but it has some merits. |
| M26 | moral_condemnation | majority | The decent majority is being held hostage by a gang of frauds. | The governing majority is being challenged by a strong opposition. |
| M27 | moral_condemnation | history | They spit on everything our grandparents fought for. | They see our history differently from how we see it. |
| M28 | moral_condemnation | adversaries | Those who oppose us are not opponents, they are enemies. | Those who oppose us are not enemies, they are opponents. |
| M29 | moral_condemnation | families | They are vultures circling over the suffering of families. | They are critics raising concerns about support for families. |
| M30 | deception_and_lies | conscience | Any person of conscience must reject these liars completely. | Any person of good will can accept parts of this proposal. |
| M31 | deception_and_lies | campaign | Their dirty tricks show how low and corrupt they really are. | Their campaign tactics show how hard they are competing for votes. |
| M32 | moral_condemnation | responsibility | They have blood on their hands and no apology can wash it off. | They made a mistake on this and they have apologised for it. |
| M33 | moral_condemnation | sin | It is a sin what they have done to the people of this land. | It is a pity what happened in the negotiations this week. |
| M34 | moral_condemnation | purity | We are pure and they are poison, and the voters know it. | We are ready and they are ready, and the voters will choose. |
| M35 | moral_condemnation | agenda | Their evil agenda must be defeated once and for all. | Their policy agenda must be debated and put to a vote. |
| M36 | deception_and_lies | sincerity | They are cynical frauds pretending to care about you. | They are serious people who care about this issue too. |
| M37 | deception_and_lies | credibility | No decent person could trust a single word from these cheats. | Most reasonable people would weigh the arguments from both sides. |
| M38 | deception_and_lies | hatred | They have poisoned our politics with lies and hatred. | They have changed our politics with new ideas and methods. |
| M39 | moral_condemnation | choice | Honest citizens against a rotten establishment, that is the choice. | Two credible parties with different programmes, that is the choice. |
| M40 | deception_and_lies | public life | They are the most dishonest people I have ever met in public life. | They are the most determined people I have ever met in public life. |

## Nuisance set: Hostility and negative affect (`hostility_negative_affect`)

Anger, fear or outrage about a situation or event, with no elite antagonist and no appeal to popular sovereignty.

| ID | Topic | Nuisance | Neutral |
|---|---|---|---|
| hos01 | floods | I am furious that these floods destroyed so many homes again this year. | I am aware that these floods affected many homes again this year. |
| hos02 | crime | This brutal attack on a young nurse is an outrage and it sickens me. | This attack on a young nurse is under investigation and I am following it. |
| hos03 | prices | The shocking rise in food prices is a disgrace that terrifies families. | The recent rise in food prices is a trend that affects many families. |
| hos04 | roads | These deadly roads are a nightmare and I am sick of the carnage. | These rural roads need repairs and I have asked for an assessment. |
| hos05 | wildfire | The horrific fires have left us angry, frightened and utterly devastated. | The recent fires have left us with repairs, reviews and new plans. |
| hos06 | unemployment | It is appalling and infuriating that the factory closure hit this town so hard. | It is important and necessary that we support this town after the factory closure. |
| hos07 | hospital | The waiting lists at this hospital are a scandal and a terrible shame. | The waiting lists at this hospital are a concern and a planning priority. |
| hos08 | drought | This cruel drought is ruining our farms and I hate watching it happen. | This long drought is affecting our farms and I am monitoring it closely. |
| hos09 | violence | Violence on our streets makes me angry and it has to stop now. | Violence on our streets has declined and we will keep working on it. |
| hos10 | accident | The mine disaster was a horrible tragedy and the grief is unbearable. | The mine incident was reviewed in detail and the report is published. |

## Nuisance set: Partisan attack on a rival party (`partisan_attack_on_opposition`)

Criticism of a named rival party's competence or record as a party competing in normal politics, without casting it as an elite set against the people.

| ID | Topic | Nuisance | Neutral |
|---|---|---|---|
| par01 | budget | The opposition party wrecked the budget with reckless spending when it was in office. | The opposition party presented a budget with different spending priorities when it was in office. |
| par02 | health | The Labour plan for health is a costly failure that will not work. | The Labour plan for health is a costly proposal that we have examined. |
| par03 | tax | Our opponents botched tax reform and left us a mess to clean up. | Our opponents proposed tax reform and we will debate the details. |
| par04 | schools | The Conservatives failed our schools for nine years with incompetent policy. | The Conservatives governed our schools for nine years with a different policy. |
| par05 | housing | The other party has no credible plan on housing and never did. | The other party has a different plan on housing and we disagree. |
| par06 | jobs | The previous government's jobs record was dismal and embarrassing. | The previous government's jobs record was mixed and is documented. |
| par07 | energy | The Greens' energy policy is naive nonsense that would raise bills. | The Greens' energy policy is an alternative that would change bills. |
| par08 | election | Our rivals ran a sloppy campaign full of empty slogans and mistakes. | Our rivals ran a campaign built on their own slogans and priorities. |
| par09 | debt | The Democrats piled up debt carelessly and cannot be trusted with money. | The Democrats increased debt to fund programmes and defend that choice. |
| par10 | infrastructure | The opposition bungled every infrastructure project it ever started. | The opposition started several infrastructure projects while governing. |

## Nuisance set: Hostility toward another state (`foreign_enemy`)

Nationalist or security hostility toward a foreign government or country, without an internal elite antagonist.

| ID | Topic | Nuisance | Neutral |
|---|---|---|---|
| for01 | border | Our neighbour's aggression on the border is a threat we will crush. | Our neighbour's position on the border is an issue we will negotiate. |
| for02 | trade | That foreign regime cheats on trade and we will punish it hard. | That foreign government differs on trade and we will discuss it. |
| for03 | security | Hostile powers want to destroy our nation and we must stand against them. | Other powers have their own interests and we must work with them. |
| for04 | territory | They stole our islands and we will never forgive that invasion. | They claim our islands and we will settle that dispute lawfully. |
| for05 | war | The enemy army committed atrocities and must be driven out by force. | The foreign army withdrew from the region under the agreed ceasefire. |
| for06 | espionage | Their spies are poisoning our country and we will expel every one. | Their diplomats are visiting our country and we will meet each one. |
| for07 | history | That empire enslaved our ancestors and still humiliates us today. | That empire ruled our ancestors and relations have changed today. |
| for08 | sanctions | We will strike back at the bullies who sanction our industries. | We will respond to the partners who review our industries. |
| for09 | migration route | A hostile neighbour is weaponising the border to attack us. | A neighbouring country is managing the border crossings with us. |
| for10 | defence | Our enemies across the sea are arming and they will regret it. | Our partners across the sea are investing and we will cooperate. |

## Nuisance set: Generic grievance (`generic_grievance`)

Complaint that life has become harder for citizens, without naming an elite responsible and without a sovereignty claim.

| ID | Topic | Nuisance | Neutral |
|---|---|---|---|
| gri01 | cost of living | Families are struggling to pay rent and cannot make ends meet anymore. | Families are paying rent and many have moved to new homes recently. |
| gri02 | pensions | Our pensioners have been left to suffer in cold homes this winter. | Our pensioners have received the heating payment this winter. |
| gri03 | youth | Young people have no future here and are forced to leave. | Young people have many options here and some choose to travel. |
| gri04 | rural | The countryside has been abandoned and its villages are dying. | The countryside has been surveyed and its villages are growing. |
| gri05 | wages | Workers have seen their wages fall year after year after year. | Workers have seen their wages change year by year by sector. |
| gri06 | health access | People wait months in pain for an operation and nobody helps them. | People wait several weeks for an operation and clinics schedule them. |
| gri07 | farmers | Our farmers are drowning in debt and losing everything they built. | Our farmers are investing in equipment and expanding what they built. |
| gri08 | towns | Towns like ours have seen shops close and streets decay for decades. | Towns like ours have seen shops open and streets repaved in recent years. |
| gri09 | energy bills | Energy bills are crushing ordinary households every single month. | Energy bills are sent to ordinary households every single month. |
| gri10 | jobs | Good jobs have disappeared and whole communities are in despair. | Good jobs have been created and whole communities are adapting. |

## Nuisance set: Incivility and personal insult (`incivility`)

Name-calling, mockery or vulgar dismissal of an individual opponent, without an elite-versus-people frame.

| ID | Topic | Nuisance | Neutral |
|---|---|---|---|
| inc01 | debate | My opponent is a clown who has no idea what he is talking about. | My opponent is a candidate who has a different view on this topic. |
| inc02 | speech | That speech was idiotic drivel from a man who cannot think straight. | That speech was a long address from a man who argued his case. |
| inc03 | minister | The minister is a pathetic fool and frankly an embarrassment. | The minister is a senior colleague and frankly an experienced one. |
| inc04 | interview | She rambled like a lunatic in that interview and made no sense. | She spoke at length in that interview and made several points. |
| inc05 | candidate | Shut up and sit down, you loser, nobody wants to hear you. | Please sit down and wait, sir, you will be heard shortly. |
| inc06 | rival | My rival is a weak, low-energy nobody who whines all day. | My rival is a careful, experienced speaker who talks all day. |
| inc07 | proposal | Only a total moron would come up with such a stupid idea. | Only a small group would come up with such a specific idea. |
| inc08 | record | He is a nasty little man with a disgusting record. | He is a younger member with a documented record. |
| inc09 | questions | That reporter asks dumb questions because she is a nasty person. | That reporter asks detailed questions because she is a careful person. |
| inc10 | leader | Their leader is clueless, lazy and dumb as a rock. | Their leader is new, busy and focused on the budget. |
