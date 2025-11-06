sensors
Article
ECG-Based Identification of Sudden Cardiac Death through
Sparse Representations
JosueR.Velázquez-González1 ,HaydePeregrina-Barreto1,* ,JoseJ.Rangel-Magdaleno2 ,
JuanM.Ramirez-Cortes2 andJuanP.Amezquita-Sanchez3
1 DepartmentofComputationalScience,NationalInstituteofAstrophysics,Optics,andElectronics,
SantaMariaTonantzintla,Puebla72840,Mexico;josueg@inaoep.mx
2 DepartmentofElectronics,NationalInstituteofAstrophysics,Optics,andElectronics,SantaMaria
Tonantzintla,Puebla72840,Mexico;jrangel@inaoep.mx(J.J.R.-M.);jmram@inaoep.mx(J.M.R.-C.)
3 FacultaddeIngeniería,UniversidadAutónomadeQuerétaro,AvRíoMoctezuma249,SanJuandelRio76807,
Mexico;jamezquita@uaq.mx
* Correspondence:hperegrina@inaoep.mx
Abstract: SuddenCardiacDeath(SCD)isanunexpectedsuddendeathduetoalossofheartfunction
andrepresentsmorethan50%ofthedeathsfromcardiovasculardiseases. Sincecardiovascular
problemschangethefeaturesintheelectricalsignaloftheheart,ifsignificantchangesarefound
withrespecttoareferencesignal(healthy),thenitispossibletoindicateinadvanceapossibleSCD
(cid:1)(cid:2)(cid:3)(cid:1)(cid:4)(cid:5)(cid:6)(cid:7)(cid:8)(cid:1)
occurrence. ThisworkproposesSCDidentificationusingElectrocardiogram(ECG)signalsanda
(cid:1)(cid:2)(cid:3)(cid:4)(cid:5)(cid:6)(cid:7)
sparserepresentationtechnique.Moreover,theuseoffixedfeaturerankingisavoidedbyconsidering
Citation: Velázquez-González,J.R.; adictionaryasaflexiblesetoffeatureswhereeachsparserepresentationcouldbeseenasadynamic
Peregrina-Barreto,H.;Rangel-
featureextractionprocess. Inthisway, theinvolvedfeaturesmaydifferwithinthedictionary’s
Magdaleno,J.J.;Ramirez-Cortes,J.M.;
marginofsimilarity,whichisbetter-suitedtothelargenumberofvariationsthatanECGsignal
Amezquita-Sanchez,J.P.ECG-Based
contains.TheexperimentswerecarriedoutusingtheECGsignalsfromtheMIT/BIH-SCDHandthe
IdentificationofSuddenCardiac
MIT/BIH-NSRdatabases.Theresultsshowthatitispossibletoachieveadetection30minbeforethe
DeaththroughSparse
SCDeventoccurs,reachingananaccuracyof95.3%underthecommonscheme,and80.5%under
Representations.Sensors2021,21,
theproposedmulti-classscheme,thusbeingsuitablefordetectingaSCDepisodeinadvance.
7666. https://doi.org/10.3390/
s21227666
Keywords:ECGsignals;sparserepresentations;suddencardiacdeath
AcademicEditors:QingZhang,
DavidSilvera-Tawil,Mahnoosh
KholghiandJuanPabloMartínez
1. Introduction
Received:2October2021 Sudden cardiac death (SCD) is an unexpected death caused by cardiovascular
Accepted:17November2021
problems [1] with or without a history of heart disease [2,3]. In general, SCD occurs
Published:18November2021
withinanhouraftertheonsetofsymptoms,althoughthepersonhasnohistoryofafatal
heartcondition[4]. SCDaccountsformorethan50%ofalldeathsfromcardiovascular
Publisher’sNote:MDPIstaysneutral
disease[1],rankingsecondastheleadingcauseofdeath,aftercancer[5]. SCDisavital
with regard to jurisdictional claims
challengeforclinicians,asitcanbeexperiencedinindividualswithnohistoryofheart
in published maps and institutional
diseases. Numerous heart diseases lead to SCD, such as ventricular tachyarrhythmias
affiliations.
(VTA),ventriculartachycardia(VT),ventricularfibrillation(VF),bradyarrhythmia(BA),
coronaryarterydiseases(CAD),valvulardiseases(RV),myocardialinfarction(MI)and
geneticfactors[6]. However,deathsbySCDarerelatedtoventriculartachyarrhythmias
(includingVFandVT)andBA[7],makingtheheartunabletopumpbloodeffectively. The
Copyright: © 2021 by the authors.
VF is an underlying quality in most SCD episodes and is considered the leading cause
Licensee MDPI, Basel, Switzerland.
andpossibledetonator[8–10],representingabout20%ofSCDepisodes. Thesurvivalrate
This article is an open access article
decreases approximately 10% per minute for patients after VF onset [1]. Therefore, an
distributed under the terms and
earlypredictionofSCDinapersonsufferingaVFisofgreatvaluefortimelyintervention,
conditionsoftheCreativeCommons
increasingthesurvivalrate.
Attribution(CCBY)license(https://
Predicting an SCD is vital since several actions can be taken to counteract it. For
creativecommons.org/licenses/by/
example,thePublicAccessDefibrillation(PAD)procedurerescuespatientsfromimpending
4.0/).
Sensors2021,21,7666.https://doi.org/10.3390/s21227666 https://www.mdpi.com/journal/sensors

Sensors2021,21,7666 2of15
deathaftercollapse. However,thesuccessrateofcardiacfunctionrestorationprimarily
dependsonwhenfirstaidisgiventostimulatetheheart[11]. Itwouldbepreferableto
preventtheonsetofSCDbyprovidingmedicalaidbeforethecollapseoccurred,which
leadstothequestionofwhetheritwouldbepossibletohavewarningsystemscapable
ofrecognizingcardiacarresthalfanhourbeforethecrisis[12]. Effortshavebeenmade
regarding this severe health problem, developing efficient ways of predicting the SCD
through invasive and non-invasive techniques [13–15]. The main goal is to predict the
SCDbeforeitsonsetusingECGsignals[13,16],sinceECGisoneofthemostimportant
physiologicalsignalstoidentifycardiacabnormalityandelectricalconductivityfeatures.
RecentworkshaveexperimentedwithfeaturesofECGandheartratevariability(HRV),
a signal extracted from ECG, to detect the subtle changes that occur within the signals
beforeanSCDandtoidentifyinadvanceapossibleSCDrisk. Also,additionalfeatures
(time,frequency,time-frequency,andnon-linear)andmachinelearningalgorithmshave
beenusedtopredictSCDfromECGandHRVsignals. Accordingtorecentreports, an
SCDcouldbepredictedupto25minbeforeitsonsetthroughintelligentsignalprocessing
methods[17,18]. Thus,toolssuchasdiagnosticsupportsystemsbasedoncomputational
analysisandsignalprocessingtechniqueshavebeenshowntohelpdetectSCDinadvance.
In [19,20], an automated prediction of SCD based on HRV signals was performed.
Signalswereanalyzedthroughtechniquesthatidentifydatarepeatabilityortime-frequency
featuressuchastheRecurrenceQuantificationAnalysis(RQA)andtheDiscreteWavelet
Transform(DWT);statisticsfeaturessuchasentropyalsowereused. Oneimportantissue
in this works is that data analysis can generate a large set of features. Then, a feature
reductionisrequiredtoconsideronlythemorerelevantfeatures;forthispurpose,some
analysissuchasKolmogorovcomplexityorfeatureranking,commonlybasedonthet-test,
areused.Toreducetheinformationthattheclassifierhastoprocessisanadvantageofthese
works. Inbothcases,apredictionupto4minbeforeSCDwasreachedthroughk-Nearest
Neighbor (kNN) and SVM (Support Vector Machine) classifiers having an accuracy of
86.8%and94.7%,respectively. Predictiontimewasincreaseduptofiveminuteswithan
accuracy of 93.71% when the kNN classifier processed time-domain features extracted
fromHRVsignals[21]. OnedisadvantageofSCDdetectionbasedonHRVsignalsisthat
computationaltimeincreases[17],whichcouldbeanissuetoconsiderinanapplication
wheretimeisrelevant. Therefore,SCDdetectionalsohasbeenstudiedbydirectlyusing
ECGsignals. In[16],theauthorsusedasimplifiedevaluationofECGsignalsbasedon
a proposed Sudden Cardiac Death Index (SCDI) for the prediction of SCD. The SCDI
integratesaweightedcombinationofthemainfeaturesidentifiedintheECGsignaland
providesawaytoobtainauniquevalue,whichisabletodifferentiatebetweennormal
andSCDclasses. TheclassificationwithSCDIandSVMreached98.68%accuracyupto
fourminutesbeforeSCD.Adifferentpredictionapproachwasproposedin[22],where
theauthorsanalyzehowECGsignalfeatureschangeinconsecutivetimeintervals. With
this analysis, the time resolution of the prediction process was increased, and, using a
multi-layer perceptron (MLP) classifier, it was possible to predict SCD 12 min before
onset. Recently, an approach to SCD prediction based on ECG signals was presented
in [18]. This approach employs the Wavelet Packet Transform (WPT), which considers
high-frequencybandsintheECGdecomposition, reachinganaccuracyof95.8%anda
prediction20minbeforeonset. However,thefrequencybandsarefixedanddependon
the sampling frequency, inhibiting the analysis of frequencies defined by the user. An
alternativewastouseEmpiricalModeDecomposition(EMD),atechniqueabletoseparate
the ECG signal into a set of frequency bands based on its information. In this way, a
prediction25minbeforeSCDwaspossible,with94%accuracy[17].
However,thesepredictionsweremadebyconsideringabinaryclassificationinnormal
andSCDsignals. ThemaindrawbackofthiscomparisonschemeisthattheECGsignalof
apatientcouldcontainfeaturesthatdifferfromanormalECGsignaldueto,forinstance,
previousheartdisease,butnotnecessarilybecauseofafutureSCDepisode. Therefore,this
evaluationcouldnotbeaccuratesincethereisahighprobabilityofSCDmisdetection.

Sensors2021,21,7666 3of15
This work addresses the feature change in the ECG signal that occurs as the SCD
eventbecomescloser,sincethiscouldhelpinearlyidentification. Amethodologybased
onsparserepresentationsallowsdistinctivefeaturestobefoundinnormalandprevious
SCDsignals. IftheseECGsignalsareanalyzedatdifferentintervalsbeforeSCD,andtheir
features are learned, a likely SCD episode could be identified in advance. The learned
dictionariesallowadynamicfeaturerepresentationofthesignaltobeobtained,providing
acertainflexibilitydegreetorecognizetheintraclassvariationandimprovethedescription
andidentificationofSCDsignals. Moreover,thisapproachconsidersanovelmulti-class
schemethatmakesitpossibletodistinguishapreviousSCDsignalfromanormalsignal
and,additionally,tomoreaccuratelyknowifthisrelatedtoacloserorfurthertimeinterval
fromtheSCD.
Followingthis,Section2containstheproposedmethod. Theexperimentsdesignedto
evaluatethefeasibilityoftheproposedmethod,andtheresultsachieved,aredescribedin
Section3. Finally,conclusionsandfutureworkareindicatedinSection4.
2. MaterialsandMethods
AblockdiagramoftheproposedmethodologyispresentedinFigure1. Asafirst
step,anautomaticdecimatedasafunctionoftime(t)isappliedtosegmenttheECGsignal;
then,theECGsignalsarenormalized. Thegenerationofthesignalbasis(dictionaries)is
performedinthetrainingphasethroughtheOMPandk-SVDalgorithms. Thetraining
enablesdictionariestolearnthemainfeaturesofeachsignalset. Thus,itisexpectedthat
thedictionarieshelptorecognizesimilaritieswithatestsignalthroughitsdecomposition.
Thecommonschemeforclassifyingpre-SCDsignalsconsistsofcomparingthefeatures
ofaninputsignal x withthefeaturesoftwosetsofsignals,normalandpre-SCD.Ifthe
featuresof xarenotsimilartothenormalsignals,then xcorrespondstoanSCDsignal.
Thisevaluationmaygeneratebiasinclassification,sinceanysignalthatdiffersfromnormal
signalswillbeassociatedwithanSCDclass. Amodificationtothecommonschemeis
proposedbyusingamulti-classevaluationofthesignal. Inthisscheme,severalclassesare
considered;forinstance,thenormalclassandsometimeintervalspre-SCD.Then,xwillbe
associatedwiththeclassofhighersimilitude. Thedifferencebetweenthetwoschemesis
illustratedinFigure2.
Figure1. FrameworkforECGsignalanalysis: pre-processingsteptoobtain1minintervalsfrom
normalandSCDsignals(yellowblock),atrainingstepforrecognizingparticularfeaturesfromthe
intervalsofinterest(blueblock),andidentificationoftestsignalsthroughtheirdecompositionby
sparserepresentations(redblock).Inthisapproach,vectorαisconsideredthefeaturevector.

Sensors2021,21,7666 4of15
Figure 2. Proposed multi-class scheme and its comparison with the common scheme for SCD
signalclassification.
2.1. Dataset
Thedataobtainedfromtwointernationalopenaccessdatabaseswereusedtoevaluate
theproposedmethodology. TheECGsignals(normalandSCD)wereobtainedfromthe
MIT/BIH Normal Sinus Rhythm (NSR) [23] and the MIT/BIH Sudden Cardiac Death
Holter(SCDH)[24]databases. InthecaseoftheNSRdatabase,ECGsignalsof18patients
areincluded. ExpertsfromtheArrhythmiaLaboratory,atBoston’sBethIsraelHospital,
confirmedthatsignalsbelongtosubjectswithahealthyheartrate,asshowninFigure3.
Ontheotherhand,theSCDHdatabaseincludestheECGsignalsof23subjectswithSCD
caused by VF; these signals were obtained from the Boston area hospitals. Each signal
contains a recording of 24 h, including the exact time of the SCD. Three recordings were
excludedbecausetheypresentedheartalterationsthatdifferedfromanSCDorVFepisode.
Figure4showsanECGsignalfromapatient2minbeforetheSCDoccurrence. Table1
summarizessomepatientfeatures. Theclinicalinformationofpatientsandthetimeof
SCDonsetareregisteredintheSCDHdatabase[24].
Figure3.ExampleofanECGsignalfromahealthysubject.
Figure4.ExampleofanECGsignalfromasubjectthatsufferedanSCDepisode.

Sensors2021,21,7666 5of15
Table1.DemographicinformationfromtheMIT/BIH-NSRandMIT/BIH-SCDHdatabases[23,24].
Gender Age
Total Male Female Unknown Range Mean
SCD 23 13 8 2 17–82 60.31
Normal 18 5 13 - 20–50 34.33
2.2. Pre-Procesing
TheECGsignalsoftheNSRandSCDHdatabaseswereacquiredat128Hzand250Hz
samplingfrequencies,respectively,anddigitizedwithananalog-to-digitalconverterof
12bits[24]. ToperformtheanalysisbetweenSCDandcontrolgroups,theECGsignalsof
theSCDgroupweredownsampledto128Hzbyconvolvingthesignalwithalow-pass
FiniteImpulseResponse(FIR)filter. SignificantSCDsymptomsgenerallyoccurwithin
onehourbeforeonset(pre-SCDsignals),eventhoughthepersondoesnothaveahistory
offatalheartcondition[4]. Sincethepre-SCDsignalshavesignificantfeaturesthatcan
beassociatedwithanSCDevent(seeFigure5),theyareusedforpredictiontasks. Then,
duringanalysisofpre-SCDsignals,thegoalistodetectsignificantchangesthatallowfor
thepredicitonofSCDusingtimeintervalsof1min[16,19–21]. Inthiswork,theminutes
5,10,15,20,25,and30beforetheSCDwereanalyzed. Additionally,the1minintervalof
thecontrolgroupisrandomlyextractedfromtheECGsignal. Allthesegments,pre-SCD
andcontrol,werenormalized,andtheirrespectiveR-Rintervalswereextracted,asshown
in Figure 6. In a 1 min interval, there are about 70 R-R intervals, since an R-R interval
lasts approximately one second; see Figure 6b. Then, there are about 1260 samples for
thenormalinterval(18subjects)and1400samplesforeachpre-SCDinterval(20subjects).
Thesesampleswereputintosetscorrespondingtoeachoftheclassesconsideredinthis
work,i.e.,C={NSR,5min,10min,15min,20min,25min,30min}. Finally,thesamples
wereanalyzedtofindtheirparticularfeaturesandclassifythem.
(a) (b) (c)
Figure5.ComparisonofsegmentsfromECGsignals:(a)normal,(b)pre-SCD,and(c)duringSCD.
(a) (b)
Figure6.(a)1-minintervalfromapre-SCDsignaland(b)anR-Rintervalextractedfromit.

Sensors2021,21,7666 6of15
2.3. SparseSignals
Asignalx n×1 ,consideredasavectorinafinite-dimensionalsubspaceRn,isstrictly
or exactly sparse if most of its entries are equal to zero, i.e., if the set of values F(x) =
{1 ≤ i ≤ n | x[i] (cid:54)= 0}isofcardinalityy (cid:28) n. Thesignalxcanbemodeledasthelinear
combinationofmelementalsignals(atoms),suchthat
m
∑
x ≈ Dα = α[i]d (1)
i
i=1
whereα m×1 isthesparserepresentationofxcontainingthecoefficientsassociatedwiththe
atoms(d i )inamatrixdictionaryD n×m involvedinthedecomposition[25,26](seeFigure7).
SignalssparsedbyDarewrittenasasuperpositionofasmallfractionoftheatomsinthe
basis.Anatomd ofn×1isanelementalsignalrepresentingpartoftheenergyorfeaturing
i
aspecifictypeofsignaltowhichthedictionarywasadapted. Thus,adictionaryDisan
indexedcollectionofmatoms,i.e.,an×mmatrix,whosecolumnsaretheatoms. Whenthe
dictionaryhasmorecolumnsthanrows,m > n,iscalledovercompleteorredundant,and
hasasettinginwhichx ≈ Dα. Twopossibleoperationscanbeperformedonadictionary:
analysisandsynthesis. Theanalysisistheoperationthatobtainsthesparserepresentation
α of a complete signal x by using the expression α = D(cid:48)x, where D(cid:48) is the transpose
dictionary. ThesynthesisperformsanapproximatereconstructionofxusingEquation(1),
asshowninFigure7.
Figure7. Reconstruction(synthesis)processofasignalxusingitssparserepresentationαanda
dictionary D. Coefficients in α are related to the atoms or elemental signals in D; therefore, an
approximationoftheoriginalsignal≈xcanbeobtained.
Inpreviousworks,overcompletedictionarieshavedemonstratedahighperformance
inclassificationtasks[27,28]. Therearetwotypesofdictionary: afixeddictionaryanda
learneddictionary. Fixeddictionariescontainpredefinedsignals,usuallygeneratedbya
knownfunction,e.g.,sineorwavelets,andprovideananalysisoperationinareasonable
processingtime. Whenthesignalstobeanalyzedhavewell-identifiedfeatures,afixed
dictionaryisthebestoption;otherwise,alearneddictionarymustbecreated. Alearned
dictionaryimpliesalearningprocessinwhichtheparticularfeaturesofasignalsetmustbe
capturedandrecognizedthroughananalysisprocess. Althoughdictionarylearningmeans
ahigherprocessingtime,thisoptionachievesabetterperformancewhenthefixedexisting
dictionariesdonotaccuratelyrepresentthesignalsthatneedtobeprocessed[27–29].
Onceadictionaryhasbeendefined,itispossibletoobtainasparserepresentationof
thesignals. Forinstance,Figure8ashowsanR-RintervaloftheoriginalECGsignal. This
signalhasbeendecomposedinatoms,alongwiththeircorrespondingcoefficients,through
theanalysisoperation. Theatomd anditsα[i]coefficientgenerateanelementalwaveform
i
thatrepresentsapartoftheoriginalsignal(seeEquation(1)). Figure8bshowssevenofthe
sixteenelementalsignalsinwhichtheoriginalsignalwasdecomposed,whileFigure8c
showsitsreconstructionthroughthesynthesisoperationbyusingadifferentnumberof

Sensors2021,21,7666 7of15
waveforms. Thehigherthenumberofsignalsusedinthereconstruction,themoresimilar
characteristicsofthereconstructedandtheoriginalsignal(Figure8d).
(a) (b)
(c) (d)
Figure8.(a)Originalsignal,(b)signaldecomposition(analysis)inelementarywaveforms,(c)signal
reconstruction(synthesis)byusingdifferentnumberofatoms,and(d)signalreconstructionwithall
numbersofatoms.
2.4. DictionaryLearning
Atraineddictionaryisobtainedthroughadictionary-learningprocess. Inthiswork,
thedictionary-learningprocessisperformedbytwoalgorithms: OrthogonalMatching
Pursuit(OMP)andk-SingularValueDecomposition(k-SVD).OMP,agreedyalgorithm,
reducestheresourcerequirementsandobtainsasparsesolutionbyperformingtheanalysis
operation,givenadictionary[25]. Afterthis,k-SVDevaluateshowaccuratethedictionary
isfordecomposingtheinputsignals. BothalgorithmsandtheiruseinSCDpredictionare
explainedindetailinthissection.
2.4.1. OrthogonalMatchingPursuit(OMP)
TheOMPalgorithmsearchesforanapproximatesolutionthroughtheselectionand
combinationofatomsinDthatminimizetheerror-constrained(Equation(2))sparsecoding
(cid:112)
problem,where(cid:107)α(cid:107) = ∑ |α |2isthe(cid:96) norm,andεistheerrorthresholdintherange
2 i i 2
[0,1]. Thus,Equation(2)allowsforsignaldecompositionuntilanεerrorlevelisreached;
therefore,thenumberofcoefficientsmayvaryfromonesignaltoanother.
α =argmin(cid:107)α(cid:107) s.t. (cid:107)x−Dα(cid:107)2 ≤ ε (2)
0 2
α
The OMP error-constrained is described in Algorithm 1, where the inputs are the
dictionary D, a signal x, and a given minimum error ε; the expected outputs are the α
vectorandaresidualr . Thealgorithmensuresthate < εandr = x,sincethesignalxwas
j 0
notyetdecomposed. I isthevectorofdimensions j×1thatstorestheindexesofatoms
involvedinthedecompositionof x. TheOMPalgorithmperformsaniterativeprocess
thatchoosestheoptimallocalsolutionfromasetofpossiblesolutions. Ineachiteration
j, thisprocesstriestofind, in D, theatom d withthehighestcorrelationtothecurrent
i

Sensors2021,21,7666 8of15
energyoftheresidualr j−1 (Algorithm1,lines3–4). Theindexofthei-thatomfulfilling
theargmaxconditionisstoredin I ateachiteration,helpingtocompilethesubmatrixD .
I
TheαvectoriscomputedwiththeatomsinD ;theresidualisupdatedasr ,containing
I j
theremainingenergyofx,whichisnotyetrepresentedbyD α(Algorithm1,lines5–6).
I
Finally,eisestimatedbytheratiobetweentheenergyremaininginr andtheenergyofthe
j
originalsignal(Algorithm1,line7). Thus,OMPgeneratesasetoflocaloptimalsolutions,
allowingittofindtheoptimalglobalsolutionforthesparserepresentationα.
Algorithm1:OrthogonalMatchingPursuit.
input :D,x,ε
output :α,r
j
initialization:I = (),r = x,e = ε+1,j =0
0
1 whilee < εdo
2 j = j+1
3 i =argmax|D(cid:48)r j−1 |
4
I(j) =i
5 α = (D I )−1x
6 r j = x−D I α
7 e = (cid:107)
(cid:107)
x
r(cid:107)
(cid:107) 2 2 ∗100
8 end
2.4.2. k-SVD
Asmentionedabove,adictionarycanbeadaptedtorecognizethecharacteristicsof
a specific type of signal. k-SVD is an algorithm that allows for the learning process to
provideabasisaccordingtoasetofsignals. Therefore,thisiscalleddictionarylearning
(Algorithm2). TheprocessstartsfromasetXcontaining Mtrainingsignalsofthesame
type,aninitialdictionaryD ,andagivennumberKofiterations;accordingtotheliterature,
0
between10and20iterationsarerequired[30,31]. Theendgoalistocapturetheessential
characteristicsofthesignalsetinafinallearneddictionaryD . First,thematrixofsparse
K
representations,α ofdimensionsm×M,isobtainedbyusingOMPandthedictionary
k
D k−1 (Algorithm2,line2). SinceOMPhasanalyzedasetofsignalsofthesametype,the
α matrixshouldcontainsomecommonatomsinthedecompositionofthesignals. Itis
k
assumedthatifanatomtakespartinthedecompositionofseveralsignals,itadequately
representspartoftheenergyofthesignalsinXandmustbepreserved;otherwise,itmust
berecomputed.Then,thematomsofD k−1 areanalyzedtofindtheirparticipationinα k .For
this,thesetofsignalswinwhichthej-thatomtakespart(Algorithm2,line4)isobtained.
Asubmatrixα isgeneratedbycontainingthewcolumnsinα andsettingitsjrowto0,
w k
withtheaimofperformingasignalreconstructionwithouttheparticipationofthe j-th
atom. Then,theresidualmatrix(R)iscomputedbythedifferenceinvaluesbetweenthe
originalsignalsetXandtheproductofthecurrentdictionarywiththecurrentcoefficients
(Algorithm2,lines5–6). Inthisway,theresidualRbetweenthesubsetoforiginalsignals
X andtheirreconstructionD α providesamoreaccurateapproximationofthej-thatom
w k w
when it is processed by SVD (Algorithm 2, line 7); where U are the eigenvalues, V the
eigenvectorsandΣthediagonalmatrixcontainingthesingularvaluesindescendingorder.
Theupdateofthej-thatomanditsrespectivecoefficient,d andα ,iscomputedinlines8–9
j j
ofAlgorithm2. Itisexpectedthat,inthefirstiterations,D k−1 doesnotprovideanaccurate
decompositionα butthatthedictionary’sabilitytorepresentXimprovesaskincreases,
k
untilitreachesD .
K

Sensors2021,21,7666 9of15
Algorithm2:K-SingularValueDecomposition.
input :D ,X,K
0
output:D
K
1 fork =1,2,...,Kdo
2 α k =OMP(D k−1 ,X,ε)
3 forj =1,2,...,mdo
4 w = {l ∈1,2,...,M | α k [j,l] (cid:54)=0}
5 α w [j,w] =0
6 R = X w −D k−1 α w
7
[UΣV] = SVD(R)
8 d j ∈ D k = u 1
9 α j ∈ α k = v 1 Σ(1,1)
10 end
11 end
3. ResultsandDiscussion
In ECG applications, it is expected that, by using α as a feature vector, the sparse
representationhelpstodistinguishbetweenthedifferentECGsignals(normalandSCD).
SincetheaimistheearlydetectionofchangesinanECGsignal,whichcouldbeassociated
withapossibleSCD,twogeneralstepswerefollowedinthismethodology: (i)dictionary
learning, to identify the features of each signal class in C and (ii) signal classification,
bymeasuringthesimilaritybetweenthefeaturesofanewinputsignalandthelearned
featuresforeachclass.
ToidentifythefeaturesofthesignalsconsideredinC,atraineddictionaryisnecessary
foreachclass.Throughthelearningprocesswithk-SVD,adictionaryidentifiesthecommon
elementalsignalsofaparticularclass.AsmentionedinSection2.2,Cconsidersthesamples
fornormalECGsignalsandsixtimeintervalsat5min,10min,15min,20min,25min,
and30minpreviousSCD,i.e.,sevenclassesintotal. Therefore,seventraineddictionaries
are required to perform ECG signal classification based on sparse representations. For
dictionarylearning,itisnecessarytohaveasetofsamplesofthesametypefromwhichthe
commonelementalsignalscanbeidentifiedthroughthek-SVDalgorithm.Forthispurpose,
thesamplesineachclassofCwererandomlyselectedanddividedintotestandtraining
subsets;thedivisionofthetrainingandtestsetsfolloweda55–45%relationship,i.e.,forthe
trainingandtestsets,tenandeightrecordingsweretakenfromtheMIT/BIHNSRdatabase,
andelevenandninerecordingsfromtheMIT/BIHSCDHdatabase. Norecordingfromthe
trainingstagewasusedfortheteststage. Thus,thek-SVDwasperformed,receivingan
initialdictionaryD filledwithrandomvalues,thetrainingsetofsamplesofaparticular
0
class c ∈ C, and K = 20 iterations as parameters. The maximum number of iterations
was set to ensure the dictionary was completely trained; fewer iterations could reduce
theperformanceduringsignaldecomposition. Asaresult,thedictionaryD ,whichwas
c
specificallyadaptedtorecognizetheelementalsignalsofclassc,isobtained. Thisprocess
isrepeatedforalltheclassesofinterest;inthiscase,foralltheclassesinC. Therefore,a
setoftraineddictionariesDT = {D ,D ,D ,D ,D ,D ,D }are
NSR 5min 10min 15min 20min 25min 30min
usedtoobtainthemostaccuratedecompositionoftheirrespectivesignals,whichcanbe
usedforsignalclassification.
Theclassofanewinputsignal x mustbeidentifiedbasedontheinformationthat
iscontainedindictionaries. Forthis,itisnecessarytoobtainadescriptionofthesignal
through its features. The α vector obtained by the sparse representation simplifies the
signalthatcanbeusedasafeaturevector. Duetothedictionary’strainingprocess,where
morerelevantelementalsignalswereselected,afeature-rankingprocessisnotnecessary.
Theαvectorcorrespondingtoxmustbeevaluatedtofindthehighersimilitudebetween
itsfeaturesandthefeaturesofaspecificsetofsignals,i.e.,adictionaryinDT. Toperform
featuresevaluation,itisnecessarytoobtainαand,tofindthehighersimilitude,xmustbe

Sensors2021,21,7666 10of15
sparsebyallthedictionariesinDT. TheOMPalgorithmisusedtosparsex(Algorithm1),
withthelearneddictionaryforaclassD ∈ DT,theinputsignaltoclassifyxandanerror
c
valueε = 0.05asparameters; ahighvalueof εlimitsthelevelofsignaldecomposition.
Thisprocessisrepeatedforeachclass; then,asetoffeaturevectorsα isobtained. For
C
classification, it is assumed that a dictionary with learned features of a particular class
mustrecognizeasignalofthesametypemoreeasilythanotherdictionaries,asreported
in[32]. Onewayofmeasuringtherecognitionofthesignalthateachdictionaryperforms
isbyassessingαcoefficients. Forinstance,ifx isasignalofclassc1,thenthedictionary
c1
D willbeabletorepresentthesignalwithoutgeneratinghighcoefficients,becausemost
c1
of the x features are already contained in the elemental signals of D . Thus, α can
c1 c1 C
beevaluatedbytheminimumsumofcoefficients,asindicatedinEquation(3),andthe
classofthei-thdictionaryisthemostlikelyclasstobeassociatedwithsignalx. Moreover,
having a trained dictionary composed of elemental signals that participate in the ECG
decompositionwithoutaspecificrankingallowsforadynamicfeatureextractionprocess.
Forexample,twosamplesbelongingtothesameclasscouldbedecomposedbycombining
differentelementalsignalsfromthedictionary. Theirenergywillbewell-representedin
theirαvectors,sincealltheelementalsignalsinthedictionarywereadaptedforthesame
typeofsignal. Inthisway,acertainflexibilityisreachedinthefeatureselection,avoiding
theuseofbothafixednumberoffeaturesandafixedranking.
∑
i =argmin |α [i]| (3)
C
C
For the classification stage, two experiments were performed under the common
schemeandthemulti-classscheme. Toguaranteethatsignalselectioninaclassification
experimentdoesnotaffectthefinalresults,atwo-foldcross-validationwascomputed,then
repeatedtentimes. Theobtainedresultsundertheproposedmethodologywereevaluated
byusingtheaccuracy(Acc)measureaspresentedinEquation(4),wheretruepositives
(TP),truenegatives(TN),falsenegatives(FN),andfalsepositives(FP)wereconsidered.
Moreover,theresultswerealsocomparedwiththoseobtainedintherelatedworks.
Accuracy(Acc): theratioofcorrectpredictionstothetotalpredictions.
TP+TN
Acc = (4)
TP+TN+FN+FP
Thesparserepresentationsofprocessedsignalsweretestedunderthecommonscheme
(Figure 9) that considers the normal and SCD signal classes. Table 2 shows the results
of one of the tests and its metrics. The results showed that, in general, the evaluation
criterion(Equation(3))couldidentifyahighersimilitudebetweentheinputsignalandits
correspondingclass,withanincreasednumberofcorrectpredictions. Theaccuracy(Acc)
indicatesthatthecorrectclassificationofpre-SCDsignalswashigherthan90%. Ageneral
evaluationconsideringtentestswasperformedtoensuretheconsistencyoftheresults.
Table 3 shows the statistics of the ten tests, where a high accuracy and low dispersion
wereobservedateachtimeinterval. Nevertheless,inthecommonschemecomparison
forthepre-SCDintervals,itislikelythatasignaldifferingfromthenormalclasswould
be detected as SCD without considering the degree of difference, i.e., lower in further
pre-SCDintervalsandhigherinthenearestpre-SCDintervals. Thisconditionmaycause
theprecisiontohaveslightvariations,despitethechangingtimeinterval.

Sensors2021,21,7666 11of15
Figure9.ThecommonschemeusedinSCDECGsignalclassification,whereaninputECGsignalis
identifiedasnormalorSCDdependingonitsfeatures;ifasignaldoesnotfitthecharacteristicsof
oneclass,thenitisassumedtobelongtotheother.
Table2.Resultsofanindividualtestunderthecommonschemme.
TimeInterval(beforeSCD) TP TN FP FN Acc(%)
5-min 382 258 55 1 92.0
10-min 375 290 62 5 90.8
15-min 369 271 68 1 90.3
20-min 373 332 64 4 91.2
25-min 383 281 54 2 92.2
30-min 397 331 40 0 94.8
Table3.MeasuresforthetentestsofECGSCDclassificationthroughsparserepresentations.
TimeInterval(beforeSCD) Acc(%)±std.dev.
5-min 94.4±2.8
10-min 93.5±2.7
15-min 92.7±3.1
20-min 94.0±3.1
25-min 93.2±3.5
30-min 95.3±2.5
A comparison with previous reports that performed pre-SCD signal classification
underthecommonschemeusingtheMIT/BIHNSRandMIT/BIHSCDHdatabasesis
presentedinTable4. Dataonthetypeofsignalprocessed,methods,classifiers,andthe
predictiontime,alongwithitsrespectiveaccuracy,werealsoincluded. Thecomparison
betweentheseapproachesandtheproposedapproachhighlightsthefactthattheECG
signalisdirectlyprocessed. OthermethodologiesusedtheHRVsignal,butthisincreases
thecomputationaltime,andacorrectionisrequiredinthedetectionofR-Rintervals[7].
Moreover,featurerankingisacommontaskinotherworks.Still,itisacomplicatedprocess,
asthebehaviorofsomefeaturesmaychangeovertime,meaningonefeatureevaluation
perminuteisneededtoidentifywhichfeaturesbetterrepresentthatspecificinterval[19].
Sincesparserepresentationsprovideasimplifieddescriptionofthesignal,αcanbeused
asfeaturevector, avoidingfeatureranking. Additionally, itwasfoundthatthenormal
andSCDsignalscanbeidentifiedwithhighprecisionusingasimplecriterioninsteadofa
moresophisticatedclassifier. Acharyaetal.[19]alsoprovedasimpleevaluationbyusing
theSuddenCardiacIndex(SCDI)todetectSCDupto4minbeforetheonset. Inprevious

Sensors2021,21,7666 12of15
works,itwasproventhatitispossibletoreachanSCDdetectionupto30minbeforeonset,
withahighaccuracy.
Table4.Predictionandaccuracycomparison.TheMIT/BIHNSRandMIT/BIHSCDHdatabaseswereusedinallcases.
Work Signal MethodsandCharacterization Classifier PredictionTime(Acc)
U.RajendraAcharyaetal. Non-linearfeaturesextracted
HRV k-NN 4minbefore(86.8%)
(2015)[19] fromDWT
U.RajendraAcharyaetal. Non-linearfeaturesextracted
ECG SVM 4minbefore(92.1%)
(2015)[16] fromDWT
M.Murugappanetal.
HRV Timedomainfeatures Fuzzy 5minbefore(93.7%)
(2015)[21]
HamidoFujitaetal. Non-linearfeaturesextracted
HRV SVM 4minbefore(94.7%)
(2016)[20] fromDWT
EliasEbrahimzadehetal. Non-linear,time-frequency,and
HRV MLP 12minbefore(88.2%)
(2018)[22] linearfeatures
Amezquita-Sanchezetal.
ECG Non-linearfeaturefromWPT EPNN 20minbefore(95.8%)
(2018)[18]
OliviaVargas-Lopezetal.
ECG Non-linearfeaturesfromEDM MLP 25minbefore(94%)
(2020)[17]
Proposed ECG SparseRepresentations Sumofabsoluteα 30minbefore(95.3%)
Althoughthetraditionalscheme(Figure9)allowsforcomparisonwiththestate-of-the-art
SCDprediction,itmightnotbesuitabletocompareonlytwoclasses: normalsignalsand
SCD signals. These SCD signals belong to patients with a history of heart disease [24].
Thus,theentiresignalmaybehavedifferentlythananormalsignal,notjustthesignalin
theminutesbeforeanSCDevent. Forthisreason,anexperimentalevaluationbasedon
multipleclasseswasperformed(Figure10). Inthiscase,theclasseswereassociatedwith
thetimeintervalsdefinedinC. SincethefeaturesoftheECGsignalchangeastheSCD
getscloser,itisassumedthat,byusingdifferentcategories,localfeatures(relatedwiththe
proximityofSCD)couldbehighlighted,whilecommonfeatures(relatedtopreviousheart
diseases)couldbeattenuated. Inthisway,theclassificationcouldbemademoresuitable.
Figure10. Proposedmulti-classschemeforSCDECGsignalclassification,inwhichisconsidered
thatdifferenceswithrespecttonormalsignaldonotnecessarilycorrespondtoanimmediateSCD
buttopre-SCDintervalsorevenotherspecificcauses;thenumberofclasses(N)dependsonthe
conditionsaddressedintheexperiment.

Sensors2021,21,7666 13of15
TestsresultsundertheproposedschemearepresentedinTable5;atwo-foldcross-validation
wascomputed,andrepeatedtentimes. Sinceaclassofnormalsignalswasincluded,an
approximationofthegeneralresultscanbecmade,usingthecommonschemeevaluated
inTable3. UnlikepreviousstudiesusingECGsignals[17,18],itcanbeseenthatthegreater
thedistancefromthestartofanSCDevent,themoredifficultitistopredicttheSCDwith
highaccuracy. FromthetentestsperformedatthetimeintervalsinC,anaverageaccuracy
of 80.5% was obtained for an SCD event up to 30 min in advance. The purpose of the
experimentalevaluationisthecomparisonofSCDsignalswiththesameconditionsofa
historyofheartdisease;therefore,thisisanevaluationwithmoreequalconditions.
Table 5. ECG SCD classification through sparse representations based on the proposed
multi-classscheme.
TimeInterval(beforeSCD) Accuracy(%)±std.dev.
Normalminute 96.3±1.4
5-min 86.2±0.9
10-min 78.4±1.6
15-min 80.1±2.1
20-min 81.0±1.3
25-min 83.8±1.0
30-min 80.5±2.8
4. Conclusions
TheearlyanticipationofSCDisvitaltomedicalspecialistswhocanapplypreventive
treatment,increasingsurvival. Itwasshownthatdictionarylearningissuitabletoaddress
ECGsignals’featureidentification,andsparserepresentationsarehelpfulasfeaturevectors.
Moreover,sincetheECGsignalissparse,throughselectingtheelementalsignalsthatbetter
representit,featurerankingwasnotnecessaryunderthisapproach. Additionally,because
signalcharacterizationwasnotbasedonfixedfeaturesbutonelementalsignals,itwas
possibletoperformafeatureextractionadaptedtothedynamicoftheECGsignals. The
experimentperformedunderthecommonschemeshowedthatthemethodologyreached
an accuracy similar or higher than related works, but by considering a wider pre-SCD
interval. However,thebinaryevaluationunderthisschemecouldbelimitedandbiasthe
classificationofpre-SCDsignals. Theproposedmulti-classschemewasabletoaddress
thedifferencesthatwerepresentamongthepre-SCDsignals,providingamoresuitable
classification. Furthermore,byconsideringthatthefeaturesoftheSCDattenuateasthe
SCDeventgetscloser,itwasexpectedthatidentificationofpre-SCDsignalswasreduced
inlongerintervalsandincreasedinshorterintervals. Thisbehaviorcorrespondswiththe
results obtained under the multi-class scheme that reached an accuracy of 80.5% up to
30minbeforeSCD.
Infuturework,ananalysisofelementalsignalsindictionarieswillbeaddressedto
identifyandfilterthosethatgeneratenoiseintheαvectorandaffectsignalclassification,
as was the case for the 5 min interval in Table 5. Since the aim is to detect an SCD
episodeinadvance,wewillseektoimplementthismethodologyasembeddedsystemfor
continuousmonitoring.
AuthorContributions:Methodologyandvalidation,J.R.V.-G.andH.P.-B.;investigation,J.R.V.-G.,
J.P.A.-S.; writing—originaldraftpreparation, J.R.V.-G.andH.P.-B.; writing—reviewandediting,
J.J.R.-M.andJ.M.R.-C.;supervision,H.P.-B.Allauthorshavereadandagreedtothepublishedversion
ofthemanuscript.
Funding:Thisresearchreceivednoexternalfunding.
InstitutionalReviewBoardStatement:Notapplicable.
InformedConsentStatement:Notapplicable.
ConflictsofInterest:Theauthorsdeclarenoconflictofinterest.

Sensors2021,21,7666 14of15
References
1. Rea,T.D.;Page,R.L. Communityapproachestoimproveresuscitationafterout-of-hospitalsuddencardiacarrest. Circulation
2010,121,1134–1140.[CrossRef]
2. Deo,R.;Albert,C.M. Epidemiologyandgeneticsofsuddencardiacdeath. Circulation2012,125,620–637.[CrossRef]
3. Fishman,G.I.;Chugh,S.S.;DiMarco,J.P.;Albert,C.M.;Anderson,M.E.;Bonow,R.O.;Buxton,A.E.;Chen,P.S.;Estes,M.;Jouven,
X.;etal. Suddencardiacdeathpredictionandprevention:ReportfromaNationalHeart,Lung,andBloodInstituteandHeart
RhythmSocietyWorkshop. Circulation2010,122,2335–2348.[CrossRef]
4. Myerburg,R.J. Cardiacarrestandsuddencardiacdeath. InHeartDisease.ATextbookofCardiovascularMedicine;ElsevierSaunders:
Philadelphia,PA,USA,1992.
5. Passman,R. Preventionofsuddencardiacdeathindialysispatients: Drugs,defibrillatorsorwhatelse? BloodPurif. 2013,
35,49–54.[CrossRef][PubMed]
6. Murukesan,L.;Murugappan,M.;Iqbal,M.;Saravanan,K. Machinelearningapproachforsuddencardiacarrestpredictionbased
onoptimalheartratevariabilityfeatures. J.Med.ImagingHealthInform.2014,4,521–532.[CrossRef]
7. Shen,T.W.;Shen,H.P.;Lin,C.H.;Ou,Y.L. Detectionandpredictionofsuddencardiacdeath(SCD)forpersonalhealthcare. In
Proceedingsofthe200729thAnnualInternationalConferenceoftheIEEEEngineeringinMedicineandBiologySociety,Lyon,
France,22–26August2007;pp.2575–2578.
8. Passman,R.;Goldberger,J.J. Predictingthefuture:Riskstratificationforsuddencardiacdeathinpatientswithleftventricular
dysfunction. Circulation2012,125,3031–3037.[CrossRef][PubMed]
9. Pagidipati,N.J.;Gaziano,T.A. Estimatingdeathsfromcardiovasculardisease:Areviewofglobalmethodologiesofmortality
measurement. Circulation2013,127,749–756.[CrossRef]
10. Zheng,Y.;Wei,D.;Zhu,X.;Chen,W.;Fukuda,K.;Shimokawa,H. Ventricularfibrillationmechanismsandcardiacrestitutions:
Aninvestigationbysimulationstudyonwhole-heartmodel. Comput.Biol.Med.2015,63,261–268.[CrossRef]
11. Aziz,E.F.;Javed,F.;Pratap,B.;Herzog,E. Strategiesforthepreventionandtreatmentofsuddencardiacdeath. OpenAccess
Emerg.Med.OAEM2010,2,99.
12. Fang,Z.;Lai,D.;Ge,X.;Wu,X. SuccessiveECGtelemetrymonitoringforpreventingsuddencardiacdeath. InProceedingsofthe
2009AnnualInternationalConferenceoftheIEEEEngineeringinMedicineandBiologySociety,Minneapolis,MN,USA,3–6
September2009;pp.1738–1741.
13. Huikuri,H.V.;Tapanainen,J.M.;Lindgren,K.;Raatikainen,P.;Mäkikallio,T.H.;Airaksinen,K.J.;Myerburg,R.J. Predictionof
suddencardiacdeathaftermyocardialinfarctioninthebeta-blockingera. J.Am.Coll.Cardiol.2003,42,652–658.[CrossRef]
14. Hallstrom,A.P.;Stein,P.K.;Schneider,R.;Hodges,M.;Schmidt,G.;Ulm,K.;CASTInvestigators. Characteristicsofheartbeat
intervalsandpredictionofdeath. Int.J.Cardiol.2005,100,37–45.[CrossRef]
15. LaRevere,M. Baroreflexsensitivityandheart-ratevariabilityinpredictionoftotalcardiacmortalityaftermyocardialinfarction.
Lancet1998,351,478–484.[CrossRef]
16. Acharya,U.R.;Fujita,H.;Sudarshan,V.K.;Sree,V.S.;Eugene,L.W.J.;Ghista,D.N.;SanTan,R. Anintegratedindexfordetection
ofsuddencardiacdeathusingdiscretewavelettransformandnonlinearfeatures. Knowl.-BasedSyst.2015,83,149–158.[CrossRef]
17. Vargas-Lopez,O.;Amezquita-Sanchez,J.P.;De-Santiago-Perez,J.J.;Rivera-Guillen,J.R.;Valtierra-Rodriguez,M.;Toledano-Ayala,
M.;Perez-Ramirez,C.A. ANewMethodologyBasedonEMDandNonlinearMeasurementsforSuddenCardiacDeathDetection.
Sensors2020,20,9.[CrossRef]
18. Amezquita-Sanchez,J.P.;Valtierra-Rodriguez,M.;Adeli,H.;Perez-Ramirez,C.A. Anovelwavelettransform-homogeneitymodel
forsuddencardiacdeathpredictionusingECGsignals. J.Med.Syst.2018,42,176.[CrossRef]
19. Acharya,U.R.;Fujita,H.;Sudarshan,V.K.;Ghista,D.N.;Lim,W.J.E.;Koh,J.E. Automatedpredictionofsuddencardiacdeathrisk
usingKolmogorovcomplexityandrecurrencequantificationanalysisfeaturesextractedfromHRVsignals. InProceedingsofthe
2015IEEEInternationalConferenceonSystems,Man,andCybernetics,HongKong,China,9–12October2015;pp.1110–1115.
20. Fujita, H.; Acharya, U.R.; Sudarshan, V.K.; Ghista, D.N.; Sree, S.V.; Eugene, L.W.J.; Koh, J.E. Sudden cardiac death (SCD)
predictionbasedonnonlinearheartratevariabilityfeaturesandSCDindex. Appl.SoftComput.2016,43,510–519.[CrossRef]
21. Murugappan,M.;Murukesan,L.;Omar,I.;Khatun,S.;Murugappan,S. Timedomainfeaturesbasedsuddencardiacarrest
predictionusingmachinelearningalgorithms. J.Med.ImagingHealthInform.2015,5,1267–1271.[CrossRef]
22. Ebrahimzadeh,E.;Manuchehri,M.S.;Amoozegar,S.;Araabi,B.N.;Soltanian-Zadeh,H. Atimelocalsubsetfeatureselectionfor
predictionofsuddencardiacdeathfromECGsignal. Med.Biol.Eng.Comput.2018,56,1253–1270.[CrossRef]
23. TheMIT-BIHNormalSinusRhythmDatabase(MIT/BIH-NSR).Availableonline:https://archive.physionet.org/physiobank/
database/nsrdb/(accessedon15November2021).
24. Greenwald,S.D. SuddenCardiacDeathHolterDatabase(MIT/BIH-SCDH).Availableonline:https://archive.physionet.org/
physiobank/database/sddb/(accessedon15November2021).
25. Beckouche,S.;Starck,J.L.;Fadili,J. Astronomicalimagedenoisingusingdictionarylearning. Astron.Astrophys.2013,556,A132.
[CrossRef]
26. Starck,J.L.;Murtagh,F.;Fadili,J.M. SparseImageandSignalProcessing:Wavelets,Curvelets,MorphologicalDiversity;Cambridge
UniversityPress:Cambridge,UK,2010.
27. Wright,J.;Ma,Y.;Mairal,J.;Sapiro,G.;Huang,T.S.;Yan,S. Sparserepresentationforcomputervisionandpatternrecognition.
Proc.IEEE2010,98,1031–1044.[CrossRef]

Sensors2021,21,7666 15of15
28. Zhao,M.;Li,S.;Kwok,J.Textdetectioninimagesusingsparserepresentationwithdiscriminativedictionaries. ImageVis.Comput.
2010,28,1590–1599.[CrossRef]
29. Valiollahzadeh,S.;Firouzi,H.;Babaie-Zadeh,M.;Jutten,C. Imagedenoisingusingsparserepresentations. InProceedingsofthe
InternationalConferenceonIndependentComponentAnalysisandSignalSeparation,Paraty,Brazil,15–18March2009;Springer:
Berlin/Heidelberg,Germany,2009;pp.557–564.
30. Aharon,M.;Elad,M.;Bruckstein,A. K-SVD:Analgorithmfordesigningovercompletedictionariesforsparserepresentation.
IEEETrans.SignalProcess.2006,54,4311–4322.[CrossRef]
31. Mairal,J.;Bach,F.;Ponce,J.;Sapiro,G.;Zisserman,A.Discriminativelearneddictionariesforlocalimageanalysis. InProceedings
ofthe2008IEEEConferenceonComputerVisionandPatternRecognition,Anchorage,AK,USA,23–28June2008;pp.1–8.
32. Díaz-Hernández,R.;Peregrina-Barreto,H.;Altamirano-Robles,L.;González-Bernal,J.;Ortiz-Esquivel,A. Automaticstellar
spectralclassificationviasparserepresentationsanddictionarylearning. Exp.Astron.2014,38,193–211.[CrossRef]

