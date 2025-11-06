Article
Improving Early Prediction of Sudden Cardiac Death Risk via
Hierarchical Feature Fusion
XinHuang1 ,GuangleJia2,*,MengmengHuang1,XiaoyuHe1,YangLi1andMingfengJiang1,*
1 SchoolofComputerScienceandTechnology(SchoolofArtificialIntelligence),ZhejiangSci-TechUniversity,
Hangzhou310018,China;xhuang@zstu.edu.cn(X.H.)
2 SchoolofInformationScienceandEngineering,HarbinInstituteofTechnology,Weihai264209,China
* Correspondence:jiaguangle2017@163.com(G.J.);m.jiang@zstu.edu.cn(M.J.)
Abstract
Suddencardiacdeath(SCD)isaleadingcauseofmortalityworldwide,witharrhythmia
servingasamajorprecursor. EarlyandaccuratepredictionofSCDusingnon-invasive
electrocardiogram(ECG)signalsremainsacriticalclinicalchallenge,particularlydueto
theinherentasymmetricandnon-stationarycharacteristicsofECGsignals, whichcom-
plicate feature extraction and model generalization. In this study, we propose a novel
SCDpredictionframeworkbasedonhierarchicalfeaturefusion,designedtocaptureboth
non-stationaryandasymmetricalpatternsinECGdataacrosssixdistincttimeintervalspre-
cedingtheonsetofventricularfibrillation(VF).First,linearfeaturesareextractedfromECG
signalsusingwaveformdetectionmethods;nonlinearfeaturesarederivedfromRRinterval
sequencesviasecond-orderdetrendedfluctuationanalysis(DFA2);andmulti-scaledeep
learningfeaturesarecapturedusingaTemporalConvolutionalNetwork-basedsequence-
to-vector(TCN-Seq2vec)model. Thesemulti-scaledeeplearningfeatures,alongwithlinear
andnonlinearfeatures,arethenhierarchicallyfused. Finally,twofullyconnectedlayers
areemployedasaclassifiertoestimatetheprobabilityofSCDoccurrence. Theproposed
methodisevaluatedunderaninter-patientparadigmusingtheSuddenCardiacDeath
Holter(SCDH)DatabaseandtheNormalSinusRhythm(NSR)Database. Thismethod
achievesaveragepredictionaccuraciesof97.48%and98.8%forthe60and30minperiods
precedingSCD,respectively. Thefindingssuggestthatintegratingtraditionalanddeep
learningfeatureseffectivelyenhancesthediscriminabilityofabnormalsamples,thereby
Received:6September2025 improvingSCDpredictionaccuracy. Ablationstudiesconfirmthatmulti-featurefusion
Revised:5October2025 significantlyimprovesperformancecomparedtosingle-modalitymodels,andvalidation
Accepted:11October2025 ontheCreightonUniversityVentricularTachyarrhythmiaDatabase(CUDB)demonstrates
Published:15October2025
stronggeneralizationcapability.Thisapproachoffersareliable,long-horizonearlywarning
Citation: Huang,X.;Jia,G.;Huang, toolforclinicalSCDriskassessment.
M.;He,X.;Li,Y.;Jiang,M.Improving
EarlyPredictionofSuddenCardiac
Keywords: SuddenCardiacDeath(SCD);asymmetric;hierarchicalfeaturefusion;electro-
DeathRiskviaHierarchicalFeature
cardiogram(ECG);Temporalconvolutionalnetwork
Fusion.Symmetry2025,17,1738.
https://doi.org/10.3390/
sym17101738
Copyright:©2025bytheauthors.
1. Introduction
LicenseeMDPI,Basel,Switzerland.
Thisarticleisanopenaccessarticle Suddencardiacdeath(SCD)constitutesaparamountglobalpublichealthchallenge,
distributedunderthetermsand characterizedbyitsunexpectednatureanddevastatinglyhighfatalityrate[1]. Theesca-
conditionsoftheCreativeCommons latingprevalenceofcardiovascularriskfactors,coupledwithdemographicagingtrends,
Attribution(CCBY)license
underscorestheurgentneedforinnovativestrategiesinpreventivecardiology[2].
(https://creativecommons.org/
licenses/by/4.0/).
Symmetry2025,17,1738 https://doi.org/10.3390/sym17101738

Symmetry2025,17,1738 2of24
The electrocardiogram (ECG) serves as the fundamental tool for non-invasive car-
diac electrical activity assessment. A critical challenge in ECG analysis stems from its
inherentasymmetricnature,whichinthiscontextreferstothecomplex,nonlinear,andnon-
stationarycharacteristicsofthesignal. Thesestatisticalasymmetriesmanifestasdynamic,
time-varying patterns—such as the differing dynamics between heart rate acceleration
and deceleration, or subtle morphological changes under stress—that are not captured
bylinearorstationarymodels. Theadvancementofportablewearablesingle-leadECG
devices has revolutionized this field, enabling cost-effective, continuous, and real-time
monitoringoutsidetraditionalclinicalsettings[3,4]. Whilestandard12-leadsystemspro-
videcomprehensivespatialcardiacinformation,thepracticaladvantagesofsingle-lead
devices—including unparalleled convenience for long-term use—make them ideal for
widespreadscreeningandambulatorymonitoring. Critically,essentialbiomarkersforSCD
riskstratification,suchasheartratevariability(HRV),canbeeffectivelycapturedfrom
high-qualitysingle-leadsignals[5].
TheevolutionofSCDpredictionmethodologieshastraversedseveraldistinctphases.
Initialresearchwaspredominantlyanchoredinmanualfeatureengineering,particularly
fromHRVtimeseries,combinedwithconventionalmachinelearningclassifiers[6]. The
subsequentparadigmintegratedadvancedsignalprocessingtechniques—suchaswavelet
transformsandempiricalmodedecomposition—toenhancefeaturequalityandmarginally
extend the prediction horizon [7]. The most recent breakthrough has been ushered in
bydeeplearning,whichdemonstratesaremarkablecapacitytoautomaticallylearndis-
criminativepatternsdirectlyfromraworminimallyprocessedphysiologicalsignals[8,9].
Forinstance,deepneuralnetworkshavenowachievedcardiologist-levelperformancein
arrhythmiadetectionfromambulatoryECGs[10].
Despitetheseprogressiveadvancements,twopivotalchallengespersistentlyimpede
clinical translation. First, the predominant focus of existing methods remains on short-
termpredictionwindows, typicallyanalyzingsignalsmereminutesbeforetheonsetof
ventricularfibrillation.However,pathophysiologicalprocessesleadingtoSCDofteninitiate
much earlier. This critical limitation severely restricts the time available for life-saving
clinicalinterventions[11]. Second,currentmethodologiesoftenoperateinmethodological
silos: theyeitherdependexclusivelyonhandcraftedfeaturesgroundedinphysiological
knowledgeorrelypurelyonend-to-enddeeplearningrepresentations. Thisdichotomy
failstofullyexploitthecomplementarystrengthsofbothapproachesincharacterizingthe
signal’sasymmetricnature,ultimatelyconstrainingmodelgeneralizationandreliabilityfor
long-horizonprediction[12].
To address these identified gaps, this paper proposes a novel hierarchical feature
fusionframeworkfortheearlypredictionofSCDrisk. Theprincipalcontributionsofthis
workarethreefold:
(1) First, we design a multi-level feature architecture that systematically and syner-
gistically integrates complementary information from linear, nonlinear, and deep
learning-basedrepresentations,therebyprovidingamulti-facetedcharacterizationof
theECGsignal’sdynamics.
(2) Second,weintroduceadedicatedhierarchicalfusionmoduletoeffectivelycombine
multi-scaletemporalcontextsextractedbyaTemporalConvolutionalNetwork(TCN-
Seq2vec)—whichisparticularlyadeptatlearningfromnon-stationarysequences—
withhandcraftedclinicalfeatures,enablingacomprehensivemodelingofasymmetric
patternsacrossdifferenttemporalscales.
(3) Third, we rigorously evaluate our model under a clinically relevant inter-patient
paradigm,demonstratingsignificantlyimprovedpredictionaccuracythroughoutan

Symmetry2025,17,1738 3of24
extended 60 min period preceding SCD onset, thereby offering a reliable tool for
long-horizonriskstratification.
Theremainderofthispaperisstructuredasfollows: Section2providesastructured
review of related work. Section 3 elaborates on the proposed methodology. Section 4
presents the experimental results and a comparative analysis. Section 5 concludes this
paperandSection6discussespotentialfutureresearchdirections.
2. RelatedWork
TraditionalFeatureExtractionandMachineLearningApproaches. EarlySCDpre-
dictionresearchprimarilyfocusedoncombiningHRVfeatureextractionwithtraditional
classifiers. Khazaeietal.[13]employednonlinearmethodssuchasincrementalentropy
forHRVfeatureextractionandachieved95%accuracy6minbeforeSCDusingclassifiers
likeDecisionTreesandSVM.Ebrahimzadehetal.[14]proposedatimelocalsubsetfeature
selection(TLSFS)method,extendingthepredictionwindowto13minpre-SCDwith84.28%
accuracy. Shietal.[15]usedEnsembleEmpiricalModeDecomposition(EEMD)forHRV
decompositionandreached94.7%accuracy14minpre-SCD.Acommonlimitationofthese
earlyapproacheswastheirrelianceonmanualfeatureengineeringandspecificclassifiers,
whichoftenledtohighcomputationalcostsandlimitedgeneralizationcapability.
AdvancedSignalProcessingandFeatureReductionMethods. Subsequentresearch
integratedadvancedsignalprocessingtechniqueswithstatisticalfeaturesanddimensional-
ityreductiontoimproveperformance. Shietal.[16]combinedDiscreteWaveletTransform
(DWT)andLocalityPreservingProjections(LPP)forfeatureprocessing,achieving97.6%
accuracy14minpre-SCDwithonly5features. Reddyetal.[17]employedHilbert-Huang
Transform(HHT)andDWT,extendingpredictionto30minpre-SCD.Centeno-Bautista
etal.[18]usedCompleteEnsembleEmpiricalModeDecompositionandachieved97.28%
accuracy 30 min pre-SCD. While these methods demonstrated improved performance,
theyfacedchallengesinhandlingthenonlinear,non-stationary,andasymmetricalnature
of physiological signals. Statistical dimensionality reduction techniques often failed to
preservetemporaldependenciesandwereinadequateforcapturingdynamic,nonlinear
relationshipswithinthedata.
DeepLearningandMulti-modalFusionApproaches. Recentyearshavewitnessed
increasedfocusondeeplearningandmulti-modalfeaturefusion. Yangetal.[19]proposed
multi-domain feature fusion, reaching 91.22% accuracy and predicting SCD 70 min in
advance—asignificantextensionofthepredictionhorizon.Telangoreetal.[20]developeda
multi-modaldeeplearningmodelwith98.81%accuracy30minpre-SCD,thoughthemodel
facedoverfittingissues. Gaoetal.[21]designedaspecializedalgorithmforlow-signal-to-
noisesingle-leadECG,achieving95.43%accuracyandproposingaSuddenCardiacDeath
Index(SCDI)forriskquantification.
ResearchGapsandOurPositioning. Despitetheseadvances,currentmethodsexhibit
severallimitationsthatourworkaimstoaddress. First,mostdeeplearningapproachesfail
toeffectivelyleveragethecomplementaryinformationprovidedbyhandcraftedfeatures
and automatically learned representations. Second, while some studies have extended
predictionwindows,maintaininghighaccuracyoverlongerhorizonsremainschalleng-
ing. Third, many models are developed under intra-patient paradigms, limiting their
clinicalapplicability.
Unlike previous work, our approach systematically integrates multi-scale features
throughadedicatedhierarchicalfusionmoduleandisevaluatedunderaclinicallyrelevant
inter-patientparadigm. Bycombiningthestrengthsoftraditionalfeatureengineeringand
deeplearning,weachieverobustperformanceacrossextendedpredictionwindowswhile
addressingthegeneralizationchallengesofexistingmethods.

Symmetry2025,17,1738 4of24
3. MaterialsandMethods
Theproposedframework(Figure1)involvesfoursequentialstages: signalprepro-
cessing,featureextraction,featurefusion,andclassification. Priortofeatureextraction,
rawECGsignalsaredenoisedusingdiscretewavelettransform(DWT)tomitigatenoise
interference. Featureextractionisdividedintothreebranches: (1)Linearfeatureextraction:
Fivelinearfeatures,includingRRintervals,QRScomplexes,andTwaves,areextracted
fromthedenoisedECGsignals. (2)Nonlinearfeatureextraction: Thescalingexponentα
1
isderivedasanonlinearfeaturefromRRintervalsequencesofeach1minECGsample
usingsecond-orderdetrendedfluctuationanalysis(DFA-2).(3)Deeplearning-basedfeature
extraction: Multiscaledeeprepresentationsareextractedfromeach1minECGsample
usingaTCNbasedsequencetovectormodel(TCN-Seq2vec). Theseheterogeneousfea-
turesarethenhierarchicallyfusedintoaunifiedrepresentationtoenhancediscriminative
power. Theclassificationstageemploysafullyconnectedlayertomapthefusedfeatures
toprobabilisticpredictionsacrosstargetcategories.
Figure1.SchematicdiagramoftheproposedSuddenCardiacDeath(SCD)predictionapproach.
3.1. DatasetDescription
Thisstudyutilizestwopubliclyavailabledatasets: theSuddenCardiacDeathHolter
Database(SCDH)[22],jointlyestablishedbyMIT-BIHandtheAmericanHeartAssociation
(AHA),andtheNormalSinusRhythm(NSR)database[23]providedbyMIT-BIH.
Table 1 provides a detailed display of the statistical information recorded in the
databaseofthedatasetusedinthisstudy. TheSCDHdatabasecontainslong-termECG
recordingsfrom23participants, withdurationsrangingfromseveralhoursto24hper
recording. These recordings capture the full progression from normal sinus rhythm to
ventricularfibrillation(VF)onset,sampledat250Hz. AstheSCDHdatawerecollected
fromdifferenthospitaldeviceswithinconsistentleadplacements,thisstudyutilizesthe
firstleadoftheprovideddual-leadsignals. Amongtheserecords,20subjectsultimately
experienced cardiac arrest. Figure 2 displays representative ECG waveforms from the
SCDHdataset. Thegreensegment(left)showstheECGwaveformpriortoVFonset,while
the red segment (right) depicts the signal after VF onset. As is evident from the figure,
thewaveformspriortotheonsetofVFexhibitaregularcardiacrhythm. Followingthe
onset,theelectricalactivityrapidlydevolvesintoachaoticandirregularpatternwithno
discerniblenormalECGcomponents.

Symmetry2025,17,1738 5of24
Table1.Detailsofthedatausedinthiswork.
Subjects
Sampling
Database Diagnosis
Rate NumberofSubjects SubjectsFeatures
13females(age20–50)
NSR Normal 128HZ 18
5males(age26–45)
8females,13males,2sex
SCD SCD 250HZ 23
unknown(age18–89)
Figure2.ECGwaveformsfromtheSCDHdataset.Thegreenandredsegmentsrepresentthesignal
priortoandafterventricularfibrillation(VF)onset,respectively.
The NSR database, serving as the control group, comprises ECG recordings from
18 healthy subjects sampled at 128 Hz. To ensure comparability and minimize analyti-
calbias, theNSRdatawereresampledto250HzusingaFastFourierTransform(FFT)-
basedspectralresamplingmethod,followedbynormalizationofbothdatasets. Thiswas
achievedbycalculatingthetargetlengthforeachsignalsegmentviatheratioofthesam-
pling rates (250/128 ≈ 1.953125). Specifically, for a standard 10 min ECG window, the
original76,800datapoints(10min×60s×128Hz)wereresampledtoanewlengthof
150,000points(10min × 60s × 250Hz). ThemethodappliesaFastFourierTransform
(FFT)totheoriginalsignal,performsoptimalinterpolationinthefrequencydomainby
zero-paddingthespectrumtothetargetlength,andthenreconstructsthesignalviathe
inverseFFT.Thisprocessensuresthepreciseconversionofthesamplingratewhileper-
fectlypreservingtheoriginaltemporaldurationofthesignalwindows,whichiscriticalfor
subsequenttime-alignedfeatureextractionandcomparativeanalysis.
Forsuddencardiacdeath(SCD)prediction,onlyECGdatasurroundingtheVFonset
areclinicallyrelevant. Therefore,thisstudyfocusesonthe60minECGsegmentspreceding
VFonset. Table2detailsthe18selectedECGrecordingsusedinthisstudy,includingthe
precisetimingofVFonsetandthestartingpointsoftheextractedsignalsegments.
Table2.DetailedrecordinformationintheSCDHdataset.
TheStartingTimeofthe
RecordName Duration VFOnsetTime
SelectedSegment
30 24:33:17 06:54:33 07:54:33
31 13:58:40 12:42:24 13:42:24
32 24:20:00 15:45:18 16:45:18
33 24:33:00 03:46:19 04:46:19
34 07:05:20 05:35:44 06:35:44
36 20:21:20 17:59:01 18:59:01
37 25:08:00 00:31:13 01:31:13
39 05:47:00 03:37:51 04:37:51

Symmetry2025,17,1738 6of24
Table2.Cont.
TheStartingTimeofthe
RecordName Duration VFOnsetTime
SelectedSegment
41 03:56:00 01:59:24 02:59:24
43 23:07:50 14:37:11 15:37:11
44 23:20:00 18:38:45 19:38:45
45 24:09:20 17:09:17 18:09:17
46 04:15:10 02:41:47 03:41:47
47 23:35:50 05:13:01 06:13:01
48 24:36:15 01:29:40 02:29:40
50 23:07:38 10:45:43 11:45:43
51 25:08:30 21:58:23 22:58:23
52 07:31:05 01:32:40 02:32:40
3.2. SignalPreprocessing
TheECGsignalpreprocessingphaseinvolvesthreesequentialsteps: segmentation,
denoising,andwaveformdetection. First,rawECGsignalsarepartitionedintodiscrete
1minintervals. Subsequently,discretewavelettransform(DWT)isappliedtoeliminate
artifacts, including baseline wander and high-frequency noise. Finally, fiducial point
detectionidentifiesfivekeyECGwaveformlandmarks: Q-wavepeak, R-wavepeak, S-
wavepeak,T-wavepeak,andT-waveend,enablingsubsequentfeatureextraction.
Tosystematicallyinvestigatethetemporalprogressiontowardventricularfibrillation
(VF),weadoptedastratifiedsegmentationapproach. ForSCDsignals,the60minpre-VF
periodwasdividedintosixconsecutive10minintervals(0–10,10–20,...,50–60min)as
illustratedinFigure3,witheachintervalsubsequentlysegmentedinto1minsamples. For
NSR(normalsinusrhythm)signals,a10minECGsegmentwasrandomlyselectedfrom
eachrecordingandpartitionedintoconsecutive1minepochs.Thisdesignensuresidentical
NSRsamplesizesacrosscomparativedatasets,therebyisolatingmodelperformancevaria-
tionstotemporalelectrophysiologicalchangesinpre-VFSCDsignals. Thestandardized
approachguaranteesobjectivecomparisonofdeteriorationpatternsprecedingventricular
fibrillation[24].
Figure3.Oursignalsegmentationmethodology.
The ECG signals from the SCDH dataset exhibit significant baseline wander, with
somerecordingsadditionallycontaminatedbyhigh-frequencynoiseinterference. Incon-
trast,NSRdatasetrecordingsdemonstratesubstantiallylowernoiselevels. Giventhese

Symmetry2025,17,1738 7of24
characteristics,weemployDWT[25]forsignaldenoisingduetoitssuperiorperformance
inanalyzinglocalsignalfeaturesandprocessingnon-stationarybiosignals. Thedenoising
procedureconsistedofthreekeysteps: (1)an8-leveldecompositionisperformedusing
Daubechies8(db8)waveletbasisfunctions,selectedfortheirmorphologicalsimilarityto
characteristic ECG waveforms. (2) the approximation coefficients at level 8 and detail
coefficientsatlevel1arezeroedtoeliminatebaselinedriftandhigh-frequencynoise,re-
spectively. (3)theECGsignalwasreconstructedusingthemodifiedwaveletcoefficients.
AsdemonstratedinFigure4,thisapproacheffectivelyremovednoiseinterferencewhile
preservingtheclinicallyrelevantmorphologicalfeaturesoftheoriginalECGsignal.
Figure4.OriginalanddenoisedECGsignals.
TheR-wavepeakpositionswereidentifiedusingthePan-Tompkinsalgorithm[26],
withimmediatecalculationoftheprecedingRRintervaluponeachR-wavedetection. Here,
the RR interval is defined as the temporal distance between the peak of the R wave in
the current cardiac beat and the corresponding peak in an adjacent beat, reflecting the
periodicityofECGsignals. CenteredoneachidentifiedR-wavepeak,weestablishasearch
windowspanning±0.25RRintervalstolocalizetheQandSpoints,whicharedetermined
asthefirstlocalminimaoccurringbeforeandaftertheR-wavepeakwithinthiswindow,
respectively. Following QRS complex detection, we implement the area-based T-wave
detectionalgorithmproposedbyZhangetal.[27]forpreciseT-wavepeaklocalization. The
algorithmoperatesthroughthefollowingprocedure:
(1) The search intervals k and k are delineated within the RR interval according to
a b
Equations(1)and(2)toensurecompleteinclusionoftheT-waveendwhilepreventing
overlapwithadjacentwaveforms.

R +[0.15RR ]+37 ifRR <220
k = i i i (1)
a
R +70 ifRR ≥220
i i

R +[0.7RR ]−9 ifRR <220
k = i i i (2)
b
R +[0.2RR ]+101 ifRR ≥220
i i i
whereRiisthei-thR-wavepeaklocationintheECGsignal.
(2) Foreachtemporalpointk,theareaintegralwithintheslidingwindowwascomputed
usingEquation(3),withthewindowwidthfixedat9samplepoints.

Symmetry2025,17,1738 8of24
k
∑
A = (s −s ) (3)
k j k
j=k−w+1
wheres representsthej-thsamplingpointoftheECGsignal,ands representsthesignal
j k
meanwithinthesmoothwindowcenteredonk.
(3) TheT-waveendwasidentifiedasthesamplepointkwithinsearchintervals[k ,k ]
a b
wheretheamplitudeA reacheditsextremum(eithermaximumorminimumdepend-
k
ingonT-wavepolarity). Subsequently,theT-wavepeakwaslocalizedbydetecting
thefirstlocalminimumwhileretrogradesearchingfromtheidentifiedendpoint.
ThefiducialpointdetectioneffectofECGsignalisshowninFigure5. Itcanbeseen
fromthefigurethatthedetectioneffectissignificant,andeachkeywaveformpointisclearly
marked,indicatingthattheabovemethodcaneffectivelyidentifyimportantfeatures.
Figure5.FiducialpointlocalizationinECGrecordings.
3.3. FeatureExtraction
3.3.1. LinearFeatureExtraction
In studies of SCD risk prediction, linear features derived from ECG signals hold
substantialclinicalvalueduetotheireaseofmeasurementandimmediateapplicabilityin
clinicalsettings. CommonlyusedlinearfeaturesincludeRRintervals,QRScomplexes,and
T-waveparameters. Forthei-thcardiaccycle(i=1,2,3,...,N),thecharacteristicfiducial
pointsaredefinedasfollows: R[i]denotestheR-wavepeakposition,Q[i]representsthe
Q-wavepeak,S[i]indicatestheS-wavepeak,T [i]correspondstotheT-wavepeak,and
peak
T [i]markstheT-waveendpoint.
peak
(1) TheprecedingRRintervalismathematicallydefinedasthetimeintervalbetweenthe
currentR-wavepeakandthepreviousR-wavepeak,andthecalculationformulais
showninEquation(4):
R[i]−R[i−1]
RR[i] = (4)
fs
Here, f isthesamplingrate.
s
(2) TheQRSdurationisdefinedasthetimeintervalbetweentheQ-wavepeakandthe
S-wavepeak,andthecalculationformulaisshowninEquation(5):
S[i]−Q[i]
QRS[i] = (5)
f
s

Symmetry2025,17,1738 9of24
(3) TheQTintervalisdefinedasthetimefromthebeginningoftheQRSwavetotheend
oftheTwave,andtheQTcintervalreferstotheQTintervalcorrectedbasedonheart
rate[28]. ThecalculationformulaisshowninEquation(6):
T [i]−Q[i] (cid:113)
QTc[i] = end · RR[i] (6)
f
s
(4) TheTp-TeintervalisformallydefinedasthetemporaldurationbetweentheT-wave
peakandT-waveendwithinthesamecardiaccycle. Thecalculationformulaisshown
inEquation(7):
T [i]−T [i]
end peak
Tp−Te[i] = (7)
f
s
(5) The amplitude of T-wave is defined as the voltage difference between the T-wave
peakandT-waveend,andthecalculationformulaisshowninEquation(8):
(cid:12) (cid:12)
T [i] = (cid:12)s(T [i])−s(T [i])(cid:12) (8)
amp (cid:12) peak end (cid:12)
Variabilityincardiaccyclecountsacross1minsamplesegmentsleadstoinconsistent
featurequantities(RRintervals,QRSdurations,QTcintervals,etc.) extractedfromdifferent
samples,whichwouldcausecompatibilityerrorsindirectmodeltraining. Tostandardize
dimensionality,wecomputethemeanvalueforeachfeaturewithineverysamplesegment.
3.3.2. NonlinearFeatureExtraction
Detrendedfluctuationanalysis(DFA)[29]isacomputationalmethodforquantifying
self-similarityandlong-rangedependenceintimeseriesdata. Thistechniquecharacterizes
fluctuationpropertiesbysystematicallyremovingtrendcomponents,demonstratingpar-
ticulareffectivenessinanalyzingsignalswithlong-termcorrelations. Thesecond-order
detrendedfluctuationanalysis(DFA2)[30]representsanenhancedversionthatemploys
quadraticpolynomialfittingtoeliminatenonlineartrends. Thisadvancedmethodprovides
more precise analysis of datasets exhibiting complex fluctuation patterns. In the DFA2
method,scalingexponentsα andα arecommonlyextractedtoquantifylong-rangecor-
1 2
relationpropertiesoftimeseriesacrossdifferentscales. Specifically,α characterizesthe
1
fluctuationbehaviorandself-similarityofthetimeseriesatsmallerscales,makingitsuit-
ablefordynamicanalysisovershortertimeintervals,whereas α describesthefluctuation
2
behaviorandself-similarityatlargerscales,therebyrevealinglong-termdependencies. The
extractionprocedurefortheDFA2α featureisasfollows:
1
(1) RRintervalextractionandpreprocessing: TheRRintervalsequenceisderivedfrom
each1minECGsignalsegment,followedbyoutlierdetection. Anyidentifiedoutliers
arereplacedwiththeaverageoftheirimmediateneighboringvalues.
(2) Cumulativesumtransformation: TheRRintervalsequence[r ,r ,...,r ]isconverted
1 2 n
intoanintegratedseriesY(k)bycomputingthecumulativesumofdeviationsfrom
theexpectedvalue,asexpressedinEquation(9):
k
∑
Y(k) = (r −r),k =1,2,...,n (9)
i
i=1
HereristhemeanoftheRRinterval.
(3) TheintegratedseriesY(k)isdividedintowindowsoflengths(wheresrangesbetween
4and16RRintervals). Thescalerangeof4to16RRintervalswasselectedforthe
DFA2analysisbasedonacombinationofphysiologicalandpracticalconsiderations.

Symmetry2025,17,1738 10of24
Physiologically,thisrange(correspondingtoapproximately4–16sforatypicalheart
rate) captures the short-term correlation properties of heart rate dynamics, which
areinfluencedbyautonomicnervoussystemregulationandhavebeenshowntobe
alteredinpathologicalstatesleadingtoSCD.Fromamethodologicalstandpoint,this
rangeiswellsuitedfortheanalysisof1minECGsegments,astheupperlimitof16
ensuresasufficientnumberofdatawindowsforarobustcalculationofthescaling
exponent,whilethelowerlimitof4excludesveryshort-scalenoise.
InconventionalDFA2,non-overlappingwindowsaretypicallyused,witheachwin-
dowindependentlyfittedbyasecond-orderpolynomialtoremovelocaltrends. However,
forshorterdatasegments,thelimitednumberofnon-overlappingwindowsmayleadto
increasedstatisticaluncertainty. Toaddressthis,amaximum-overlapwindowingstrategy
isadoptedinthisstudytoenhancethenumberofanalyzablesubintervals. Specifically,for
agivenwindowlengths,eachnewwindowstartsonesamplepointafterthepreviouswin-
dow’sstartingposition,resultinginanoverlaplengthofs−1betweenadjacentwindows.
(4) Withineachsubinterval,thelocaltrendy(i)isestimatedviasecond-orderpolynomial
fitting. ThedetrendedfluctuationF(v,s)isthencomputedastheroot-mean-square
deviationoftheintegratedseriesfromthefittedtrend,asgivenbyEquation(10):
(cid:115)
F(v,s) =
1∑ s
[Y(v+i−1)−y(i)]2 (10)
s
i=1
wherevdenotesthestartingindexofthewindow. ThefluctuationfunctionF(s)isobtained
byaveragingF(v,s)acrossallwindows. Finally,thescalingexponentα isderivedthrough
1
log-logregressionanalysis.
Figure 6a,b present the visualization results of scale exponent α derived from RR
1
intervalsequencesusingDFA2andtheboxplotsofDFA2α featuresfordifferentsample
1
categoriesacrosssixdatasets,respectively.
Figure6.Nonlinearfeatureextraction.
Asevidencedbythefigure,theα valuesofNSRsamplesaredistributedintherange
1
of1.35±0.2,whilethoseofSCDsamplesfallwithin0.95±1.5,demonstratingasignificant
differenceinDFA2α valuesbetweennormalECGsignalsandSCDsignals,wherelower
1
valuesindicatehigherSCDrisk.
ThesuccessfuldiscriminationachievedbytheDFA2α featureunderscoresitsutility
1
inquantifyingthenonlinear,long-rangecorrelationswithintheheartrate. Thesecomplex
correlationpatternsareadirectmanifestationofthedynamicasymmetriesintheautonomic
nervoussystem’sregulationoftheheart. Specifically,adecreaseinα towards0.5(uncor-
1
relatednoise)suggestsabreakdowninthehealthy,fractal-liketemporalstructureofthe
heartbeat—akeyaspectofthepathologicalasymmetrythatourmodelaimstocapture.

Symmetry2025,17,1738 11of24
3.3.3. DeepLearning-BasedFeatureExtraction
We employ a Temporal Convolutional Network-based sequence-to-vector model
(TCN-Seq2vec)fordeepfeatureextraction,whosearchitectureisillustratedinFigure7a.
Themodelisdesignedtohierarchicallycapturemulti-scaletemporalpatternsfromECG
signalsandcondensetheentiresequenceintoaninformativefixed-lengthvector. Thisis
achievedthroughthesynergisticoperationofmultipleTemporalConvolutionalBlocks
(TCNBlocks) for feature extraction and a Temporal Attention Mechanism for dynamic
featureaggregation.
Figure7.ArchitectureoftheTCN-Seq2vecModel.
Thespecificstructureandparametersofthemodelareasfollows:
(1) Temporal Convolutional Block (TCNBlock). As the fundamental building unit,
eachTCNBlockisdesignedtocapturetemporaldependenciesataspecificscale. Theuse
ofdilatedconvolutionsiscrucialforexponentiallyexpandingthereceptivefieldwithout
losingresolutionorincreasingcomputationalcostexcessively. Thisallowsthenetwork
tointegrateinformationfrombothimmediateanddistanttimepointsintheECGsignal,
whichisessentialforcapturingbothtransientarrhythmiceventsandlonger-termheart
ratetrends. Specifically,eachblockimplements:
Dilated Convolutions: Using a fixed kernel size of 3 with dilation rates following a
geometricprogression(1,2,4)acrossconsecutiveblocks,enablingexponentialreceptivefield
expansionfromapproximately5to85timestepswhilemaintainingcomputationalefficiency.
NormalizationandRegularization: Eachconvolutionallayerisfollowedbyweight
normalization,ReLUactivation,anddropoutregularization(rate=0.5)toensuretraining
stabilityandpreventoverfitting.
Residual Connections: Identity or 1 × 1 projection shortcuts are incorporated to
mitigategradientvanishing,formulatedas:
Output = ReLU(Residual(x)+Block(x))
whereBlock(x)representsthesequentialprocessingthroughthetwonormalizedconvolu-
tionallayers.

Symmetry2025,17,1738 12of24
(2)TemporalAttentionMechanism. Thismoduledynamicallyidentifiesandempha-
sizesclinicallysalientsegments(e.g.,specificarrhythmicbeatsorischemicmorphologies)
bycomputingadaptiveweightsfordifferenttimesteps,therebyproducingacontext-aware
vectorrepresentationwherecriticalperiodscontributemoresignificantly. Itsimplementa-
tioninvolvesthreededicatedcomponents:
Feature Reduction. The high-dimensional feature maps from the final TCNBlock
(128channels)arefirstprocessedbyadedicatedTCNBlockconfiguredwithakernelsizeof
3,adilationrateof1,and64outputchannels. Thisstepreducesthefeaturedimensionality,
servingasaninformationbottleneckthatfocusesthesubsequentcomputationonthemost
salientpatterns. Theresultingtensoristhenpermutedfromtheshape(batch,channels,
timesteps)to(batch,timesteps,channels)toalignthetemporaldimensionforattention
scoring.
AttentionWeightComputation. Thepermutedfeaturesarefedintoatwo-layerfully
connectednetworktocomputetheimportanceofeachtimestep.Thefirstlinearlayer(64→
32dimensions)employsaTanhactivationfunctiontocapturenon-linearinteractions,while
thesecondlinearlayer(32→1dimension,withoutbias)producesascalarimportancescore
foreachtimestep. Theserawscoresarethennormalizedacrossthetemporaldimension
viaaSoftmaxfunctiontogeneratethefinalattentionweightsα,whichsumtoone.
i
Context-AwareAggregation. Thefinaloutputiscomputedasaweightedsumofthe
original(reduced)featuresequence,formallyexpressedasz = ∑T α h,whereh isthe
i=1 i i i
featurevectoratthei-thtimestep. Thisoperationaggregatestheentirevariable-length
sequence into a single, fixed-dimensional vector z, whose representation is dominated
bythefeaturesfromtimestepsdeemedmostcriticalforSCDrisk,therebyenhancingthe
model’sdiscriminativepowerandpotentialclinicalinterpretability.
(3)TCN-Seq2vecModel. AsshowninFigure7a,themodelfirstprocessestheinput
sequencethroughmultipleTCNBlockswithincreasingdilationrates(1,2,4)andchannel
numbers (32, 64, 128). This design creates a hierarchical feature pyramid, where lower
layers capture fine-grained, short-term patterns (e.g., spike shapes) and higher layers
capturecoarse-grained,long-termtrends(e.g.,heartrateinstability). Theoutputsfrom
theseblocksarethenfused. Finally,theTemporalAttentionMechanismisappliedtothis
multi-scalerepresentation,transformingthevariable-lengthsequenceintoadiscriminative
fixed-length vector for the final classification, effectively summarizing the entire 1 min
ECGepisode.
Crucially,thecombinationofmulti-scaletemporalconvolutionanddynamicattention
empowerstheTCN-Seq2vecmodeltoeffectivelyhandletheasymmetric,non-stationary
characteristics of ECG signals. It does not assume signal stability but instead learns to
identify and weigh the importance of critical, often short-lived, pathological episodes
withinthelongercontext,whichisessentialforearlySCDprediction.
3.4. HierarchicalFeatureFusionandClassification
Acentralhypothesisofthisworkisthattheeffectiveintegrationofcomplementaryin-
formationfromheterogeneousfeatures—rangingfromclinicallyinterpretablehandcrafted
featurestohigh-dimensionaldeeprepresentations—ispivotalforrobustandaccurateSCD
prediction. ThisisbecausethepathologicalprocessesleadingtoSCDmanifestascomplex
asymmetriesacrossmultipleaspectsoftheECGsignal. Whileexistingmethodsoftenrely
onasingletypeoffeatureoremploysimpleconcatenation,theyfailtocapturetheintrinsic,
hierarchicalnatureofcardiacelectrophysiologicalpatternsthatmanifestacrossdifferent
temporalscales.Forinstance,handcraftednonlinearfeatures(likeDFA2)explicitlyquantify
oneformofdynamicasymmetryinheartrate,whiledeeplearningfeaturesimplicitlylearn
anotherfromrawwaveformmorphology. Asimpleconcatenationcannotfullymodelthe

Symmetry2025,17,1738 13of24
interactionsbetweenthesedifferentcharacterizationsofasymmetry. Tobridgethisgap,we
proposeadedicatedHierarchicalFeatureFusionModule,thecoreofwhichisillustratedin
Figure8.
Figure8.HierarchicalFeatureFusionModuleandtheClassifier.
The design of this module is driven by the need to preserve and synergize multi-
scalecontextualinformation. Theprocessbeginsbyharvestingdeepfeaturemapsfrom
successiveTCNBlocks. Eachoftheseblocksinherentlycapturestemporaldependencies
at a unique scale; shallow blocks extract local, short-term patterns (e.g., morphological
nuancesofindividualbeats),whiledeeperblocksmodelmoreglobal,long-termcontext
(e.g.,heartratetrendoverminutes). Toconvertthesevariable-lengthfeaturemapsintoa
fixed-dimensionalyetinformativerepresentation,weapplyGlobalAveragePooling(GAP)
after each TCNBlock. This operation not only standardizes the feature dimensions but
also enhances translation invariance and mitigates overfitting by reducing the number
ofparameters.
The resulting vectors (32-, 64-, and 128-dimensional) encapsulate the hierarchical
characteristicsoftheinputsignal. Tofacilitateacoherentfusion,thesemulti-scalevectors
are then projected into a unified 64-dimensional semantic space using dedicated 1 × 1
convolutionallayers. Thiscrucialstepservestwopurposes: firstly,itachievesdimensional
alignment for subsequent processing, and secondly, it performs a learnable, non-linear
transformation that allows the model to calibrate and weight the contribution of each
featurescaleadaptively. Thisisasignificantadvancementovernaiveconcatenation,asit
enablesfeatureinteractionandrefinementbeforethefinalfusion.
Thefinalstepinvolvestheconcatenationofthealignedmulti-scaledeepfeatureswith
thehandcraftedlinearandnonlinearfeatures. Thiscreatesacomprehensiveandenriched
representationthatleveragesboththeabstract,high-levelpatternslearnedbytheTCNand
thedomain-specific,clinicallygroundedknowledgeembeddedinthehandcraftedfeatures.
Thisfusedfeaturevectoristhenpassedthroughatwo-layerfullyconnectedclassifierwith
ReLUactivationanddropoutregularizationforthefinalSCDriskprediction. Theentire
hierarchicalfusionstrategyensuresthatthemodel’sdecisionisinformedbyarichtapestry
ofinformationspanningfromlocalsignaldetailstoglobalphysiologicaltrends,thereby
significantlyenhancingitsexplanatorypowerandpredictivereliability.

Symmetry2025,17,1738 14of24
4. ExperimentsandResults
4.1. ImplementationDetailsandEvaluationMetrics
4.1.1. ImplementationDetails
The software and hardware environments used in this study are as follows: the
operatingsystemisUbuntu20.04.6LTS(Canonical,London,UK),theNVIDIAgraphics
cardisNVIDIAGeForceRTX4090(NVIDIAInc.,SantaClara,CA,USA),thedeeplearning
frameworkisPyTorch2.1.0(MetaInc.,MenloPark,CA,USA),andtheparallelcomputing
platformisCUDA12.1(NVIDIAInc.,SantaClara,CA,USA).Duringthemodeltraining
process, the cross-entropy loss function and the Adaptive Moment Estimation (Adam)
optimizerwereadopted. Theinitiallearningratewassetto0.001,anditwasreducedbya
factorof0.1afterevery5trainingepochs. Thebatchsizeofthemodelwassetto32,and
themaximumnumberoftrainingepochswassetto40toensurethestabilityofthemodel.
Forlinearfeatureextraction, theprocessinvolvedcomputing5-dimensionalhand-
craftedfeaturesfromECGsignals,includingtime-domain,statistical,andmorphological
characteristics. ThesefeatureswerenormalizedusingStandardScalertoensurezeromean
andunitvariance. Keyhyperparametersincludeda15,000-pointanalysiswindowwithno
overlapandphysiologicallyplausibleheartrateconstraints.
Forthenonlinearfeatureextraction,DetrendedFluctuationAnalysis(DFA)wasem-
ployedtocapturelong-rangecorrelationsinheartratevariability,yieldinga1-dimensional
scalingexponent. TheDFAimplementationrequiredcarefulparameterselectionincluding
logarithmicscalingranges(4–16forshort-termand16–64forlong-termfluctuations),50%
segment overlap, and first-order detrending. Additional non-linear measures such as
sampleentropywereconsideredwithembeddingdimension2andtolerancethreshold0.2.
FortheTCN-Seq2vecdeepfeatureextractionmodule,atemporalconvolutionalnet-
workwithmulti-scaleprocessingautomaticallylearnedhierarchicalrepresentationsfrom
rawECGsignals. TheTCNarchitectureemployedprogressivelyincreasingchannels(32,
64,128)withexponentialdilationfactors(1,2,4). Atemporalattentionmechanismwith
reduction to 64 channels and tanh activation highlighted clinically relevant segments.
Comprehensiveregularizationincludedspatialdropout(0.5),featuredropout(0.3),and
attentiondropout(0.2)topreventoverfitting.
4.1.2. EvaluationMetrics
Tocomprehensivelyevaluatetheperformanceoftheproposedmethod,precision(Pre),
recall(Rec),accuracy(Acc),andF1-score(F1)wereselectedastheevaluationmetrics. Their
calculationformulasareshowninEquations(11)to(14):
TP
Pre= ×100% (11)
TP+FP
TP
Rec= ×100% (12)
TP+FN
TP+TN
Acc= ×100% (13)
TP+TN+FP+FN
(Pre×Rec)
F =2× (14)
1 (Pre+Rec)
Here,TruePositive(TP)andTrueNegative(TN)representthenumberofcorrectly
classifiedpositivesamplesandnegativesamples,respectively;FalsePositive(FP)andFalse
Negative(FN)representthenumberofincorrectlyclassifiedpositivesamplesandnegative
samples,respectively.

Symmetry2025,17,1738 15of24
4.2. ExperimentalResult
To evaluate the generalization capability of the proposed method, we conducted
10independentrepeatedexperimentsforeachofthesixdatasetsthatweobtainedduring
thesignalpreprocessing(SCD10,SCD20,SCD30,SCD40,SCD50,SCD60). Ineachexper-
iment,thedatasetwasrandomlypartitionedintotrainingandtestingsetsfollowingan
inter-patientparadigm,withspecificrecordnumbersdetailedinTable3(whereeachrecord
numbercorrespondstoadistinctpatient).Tosystematicallypresentthefindings,Tables4–9
providedetailedexperimentalresultsforsixdatasets.
Table3.RecordIdentifiersforTrainingandTestSets.
ExperimentID TrainingSet(RecordNo.) TestSet(RecordNo.)
1 30,32,34,36,37,41,43,44,45,46,47,51,52 31,33,39,48,50
2 30,31,33,34,36,39,41,43,44,45,48,50,51 32,37,46,47,52
3 30,31,32,33,34,36,37,41,43,44,45,46,48 39,47,50,51,52
4 30,33,37,39,41,43,45,46,47,48,50,51,52 31,32,34,36,44
5 31,32,33,34,39,43,44,45,46,47,48,51,52 30,36,37,41,50
6 30,31,32,34,36,37,41,43,44,46,48,50,51 33,39,45,47,52
7 30,31,32,33,36,39,43,44,45,47,48,51,52 34,37,41,46,50
8 30,31,32,33,37,41,45,46,47,48,50,51,52 34,36,39,43,44
9 30,31,32,33,34,36,37,39,44,45,46,47,48 41,43,50,51,52
10 30,31,32,33,34,36,37,41,43,45,47,48,51 39,44,46,50,52
Table4.ExperimentalResultsontheSCD10Dataset.
ExperimentID TP FN FP TN Rec Pre Acc F
1
1 50 0 0 50 100.00% 100.00% 100.00% 100.00%
2 50 0 1 49 99.00% 99.02% 99.00% 99.00%
3 50 0 1 49 99.00% 99.02% 99.00% 99.00%
4 50 0 1 49 99.00% 99.02% 99.00% 99.00%
5 49 1 0 50 99.00% 99.02% 99.00% 99.00%
6 50 0 1 49 99.00% 99.02% 99.00% 99.00%
7 49 1 1 49 98.00% 98.00% 98.00% 98.00%
8 50 0 1 49 99.00% 99.02% 99.00% 99.00%
9 49 1 0 50 99.00% 99.02% 99.00% 99.00%
10 49 1 0 50 99.00% 99.02% 99.00% 99.00%
Mean — — — — 99.00% 99.02% 99.00% 99.00%
Table5.ExperimentalResultsontheSCD20Dataset.
ExperimentID TN FP FN TP Rec Pre Acc F
1
1 50 0 0 50 100.00% 100.00% 100.00% 100.00%
2 50 0 0 50 100.00% 100.00% 100.00% 100.00%
3 50 0 3 47 97.00% 97.17% 97.00% 97.00%
4 50 0 1 49 99.00% 99.02% 99.00% 99.00%
5 50 0 1 49 99.00% 99.02% 99.00% 99.00%
6 50 0 3 47 97.00% 97.17% 97.00% 97.00%
7 50 0 0 50 100.00% 100.00% 100.00% 100.00%
8 50 0 0 50 100.00% 100.00% 100.00% 100.00%
9 49 1 0 50 99.00% 99.02% 99.00% 99.00%
10 50 0 1 49 99.00% 99.02% 99.00% 99.00%
Mean — — — — 99.00% 99.04% 99.00% 99.00%

