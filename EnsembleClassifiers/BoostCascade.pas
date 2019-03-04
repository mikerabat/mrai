// ###################################################################
// #### This file is part of the artificial intelligence project, and is
// #### offered under the licence agreement described on
// #### http://www.mrsoft.org/
// ####
// #### Copyright:(c) 2014, Michael R. . All rights reserved.
// ####
// #### Unless required by applicable law or agreed to in writing, software
// #### distributed under the License is distributed on an "AS IS" BASIS,
// #### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// #### See the License for the specific language governing permissions and
// #### limitations under the License.
// ###################################################################

unit BoostCascade;

// ##########################################################
// #### Learning a cascade of classifiers
// #### according to Viola, Jones: Robust Real-Time Face Detection
// ##########################################################

interface

uses SysUtils, BaseClassifier, BaseMathPersistence, CustomBooster, RandomEng, Types;

type
  TCustomBoostArr = Array of TCustomBoostingClassifier;

type
  TCascadeBoostClassifier = class(TCustomClassifier)
  private
    fCascade : TCustomBoostArr;
    fRefClassVal : integer;

    fActIdx : integer;
  protected
    class function ClassIdentifier : String; override;
    procedure DefineProps; override;
    function PropTypeOfName(const Name : string) : TPropType; override;

    procedure OnLoadIntProperty(const Name : String; Value : integer); override;
    procedure OnLoadBeginList(const Name : String; count : integer); override;
    function OnLoadObject(Obj : TBaseMathPersistence) : boolean; override;
    procedure OnLoadEndList; override;
  public
    function Classify(Example : TCustomExample; var confidence : double) : integer; override;

    constructor Create(cascade : TCustomBoostArr; RefClassVal : integer);
    destructor Destroy; override;
  end;

type
  // can be used to update the clasifiers properties!
  TLearnerCreate = procedure(Sender : TObject; cl : TCustomWeightedLearner) of Object;

type
  TCascadeBoostProps = record
    // base properties
    PruneToLowestError : boolean;
    InitClassSpecificWeights : boolean;
    WeakLearner : TCustomWeightedLearnerClass;

    learnerClass : TCustomBoostLearnerClass;
    OverallFalsePosRate : double;
    maxNumCascade : integer;

    featureIncrease : double;    // if lower than 1 - in percent of the size of the last detector.
                                 // if higher -> fixed number of features are added if the previous cascade element
                                 //              already contains of at least this number of features

    validationPerc : double;     // value 0-1 - Percentage of examples used to create the validation set

    // init properties: define the first N classifiers according to these props:
    minFalsePosRate : TDoubleDynArray;
    minDetectRate : TDoubleDynArray;
    NumFeatures : TIntegerDynArray;
  end;

type
 TCascadeBoostLearner = class(TCustomLearner)
 private
   fProps : TCascadeBoostProps;
   fValidationSet : TCustomLearnerExampleList;
   fTrainSet : TCustomLearnerExampleList;
   fLearnerCreate : TLearnerCreate;
   fRndEng : TRandomGenerator;
   fNumCascade : integer;
   fLastProgress : integer;
   fWeakLearnerCreate: TLearnerCreate;

   procedure AdjustBOnValidationSet(cl : TCustomClassifier; const desiredD : double; var F, D : double);
   procedure CreateValidationSet;
   procedure OnWeakClProgress(Sender : TObject; progress : integer);
 protected
   function DoUnweightedLearn : TCustomClassifier; override;
 public
   property OnLearnerCreate : TLearnerCreate read fLearnerCreate write fLearnerCreate;
   property OnWeakLearnerCreate : TLearnerCreate read fWeakLearnerCreate write fWeakLearnerCreate;
   procedure SetProperties(const Props : TCascadeBoostProps);

   class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
   
   constructor Create;
   destructor Destroy; override;
 end;

implementation

uses SimpleDecisionStump, AdaBoost, contnrs, math, MathUtilFunc;

const cBoostCascadeRefClassVal = 'boostCascadeRefCl';
      cBoostCascadeCascade = 'boostCascadeArr';

{ TCascadeBoostLearner }

type
  THackBoostCl = class(TCustomBoostingClassifier);

procedure TCascadeBoostLearner.AdjustBOnValidationSet(cl: TCustomClassifier; const desiredD : double; var F, D: double);
var counter : integer;
    classVal : integer;
    conf : TDoubleDynArray;
    exmplClass : integer;
    numPos : integer;
    numNeg : integer;
    numdectRateExmpls : integer;
    maxThresh : double;
    minThresh : double;
    iter : double;
begin
     F := 0;
     D := 0;
     numPos := 0;
     numNeg := 0;
     SetLength(conf, fValidationSet.Count);

     // D: detection rate of class val = 1.
     // F: false positive rate
     for counter := 0 to fValidationSet.Count - 1 do
     begin
          classVal := cl.Classify(fValidationSet.Example[counter]);
          exmplClass := fValidationSet.Example[counter].ClassVal;

          if exmplClass = 1 
          then
              inc(numPos)
          else
              inc(numNeg);
          
          if (classVal = exmplClass) and (exmplClass = 1)
          then
              D := D + 1
          else if (classVal = 1) and (exmplClass <> 1) 
          then
              F := F + 1;
     end;

     // set B such that the desired detection rate is met
     numdectRateExmpls := Min(numPos, Round(desiredD)*numPos);

     if D < numdectRateExmpls then
     begin
          // ###############################################
          // #### "binary" search for a threshold that allows the anticipated detection rate

          // find the maximum threshold possible
          minThresh := 0;
          for counter := 0 to Length(THackBoostCl(cl).fWeights) - 1 do
              minThresh := minThresh + THackBoostCl(cl).fWeights[counter];
           
          minThresh := -minThresh; 
          maxThresh := THackBoostCl(cl).fB;

          iter := (maxThresh - minThresh)/1000;
          
          // sort the list to ease the detection rate search
          while (maxThresh > minThresh) do
          begin
               // try as best as we can
               THackBoostCl(cl).fB := (maxThresh + minThresh)/2;

               F := 0;
               D := 0;
               // reevaluate validation set -> Get the real rates
               for counter := 0 to fValidationSet.Count - 1 do
               begin
                    classVal := cl.Classify(fValidationSet.Example[counter]);

                    exmplClass := fValidationSet.Example[counter].ClassVal;
  
                    if (classVal = exmplClass) and (exmplClass = 1)
                    then
                        D := D + 1
                    else if (classVal = 1) and (exmplClass <> 1) 
                    then
                        F := F + 1;
               end;

               if D < numdectRateExmpls 
               then
                   minThresh := THackBoostCl(cl).fB 
               else
                   maxThresh := THackBoostCl(cl).fB;

               // ensure that we get not lost in the search 
               maxThresh := maxThresh - iter;
               minThresh := minThresh + iter;
          end;
     end;

     // we have found a valid classifier with adjusted B
     // -> adjust the weights according to Lienhardt et al such that the overall
     // rate is increased
     // todo!

     if numPos > 0 
     then
         D := D/numPos
     else
         D := 0;
     if numNeg > 0 
     then
         F := F/numNeg
     else
         F := 0;
end;

class function TCascadeBoostLearner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := True;
end;

constructor TCascadeBoostLearner.Create;
begin
     fProps.PruneToLowestError := True;
     fProps.InitClassSpecificWeights := True;
     fProps.WeakLearner := TDecisionStumpLearner;
     fProps.learnerClass := TDiscreteAdaBoostLearner;
     fProps.OverallFalsePosRate := 1e-3;
     fProps.featureIncrease := 25;
     fProps.validationPerc := 0.2;

     fRndEng := TRandomGenerator.Create;
     fRndEng.RandMethod := raMersenneTwister;
     fRndEng.Init(0);

     inherited Create;
end;

procedure TCascadeBoostLearner.CreateValidationSet;
begin
     fTrainSet.Free;
     fValidationSet.Free;
     DataSet.CreateTrainAndValidationSet( round(fProps.validationPerc*100), fTrainSet, fValidationSet );
end;

function TCascadeBoostLearner.DoUnweightedLearn: TCustomClassifier;
var cascade : TCustomBoostArr;
    F, D : double;  // actual False positive (F) and Detection (D) rate
    prevF : double;
    ni : integer;
    boostProps : TBoostProperties;
    minFalsePosStageRate : double;
    learner : TCustomBoostLearner;
    counter: Integer;
    featureIncrement : integer;
    numRounds : integer;
begin
     // set at least one default false positive rate
     if Length(fProps.minFalsePosRate) = 0 then
     begin
          SetLength(fProps.minFalsePosRate, 1);
          fProps.minFalsePosRate[0] := fProps.OverallFalsePosRate;
     end;
     if Length(fProps.minDetectRate) = 0 then
     begin
          SetLength(fProps.minDetectRate, 1);
          fProps.minDetectRate[0] := 0.99;
     end;

     SetLength(cascade, fProps.maxNumCascade);

     F := 1.0;
     D := 1.0;
     fNumCascade := 0;

     while (dataSet.Count > 2) and (F > fProps.OverallFalsePosRate) and ((fProps.maxNumCascade <= 0) or (fNumCascade < fProps.maxNumCascade)) do
     begin
          CreateValidationSet;
          
          ni := 1;
          if fNumCascade < Length(fProps.NumFeatures) then
             ni := fProps.NumFeatures[fNumCascade];

          minFalsePosStageRate := fProps.minFalsePosRate[Min(Length(fProps.minFalsePosRate) - 1, fNumCascade)];
          prevF := F;

          if fProps.featureIncrease < 1 then
          begin
               featureIncrement := ni;
               if fNumCascade > 0 then
                  featureIncrement := Round(cascade[fNumCascade - 1].Classifiers.Count*fProps.featureIncrease);
          end
          else
              featureIncrement := Round(fProps.featureIncrease);

          cascade[fNumCascade] := nil;
          
          // adjust the numbers of features until a specific false positive rate is achivied
          // try only two times - more increase is not feasable
          numRounds := 0;
          while ( (F > minFalsePosStageRate*prevF) and (dataSet.Count > 3) ) and (numRounds < 3) do
          begin
               cascade[fNumCascade].Free;
               
               boostProps.NumRounds := ni;
               boostProps.PruneToLowestError := fProps.PruneToLowestError;
               boostProps.InitClassSpecificWeights := fProps.InitClassSpecificWeights;

               boostProps.OwnsLearner := False;
               boostProps.WeakLearner := fProps.WeakLearner.Create;
               boostProps.WeakLearner.OnLearnProgress := OnWeakClProgress;

               if Assigned(fWeakLearnerCreate) then
                  fWeakLearnerCreate(self, boostProps.WeakLearner);
               
               // #######################################################
               // #### learn the classifier
               learner := fProps.learnerClass.Create;
               try
                  learner.SetProperties(boostProps);
                  learner.Init(fTrainSet);
                  
                  // step to update learner properties!
                  if Assigned(fLearnerCreate) then
                     fLearnerCreate(Self, learner);

                  cascade[fNumCascade] := learner.Learn as TCustomBoostingClassifier;
               finally
                      learner.Free;
               end;
               boostProps.WeakLearner.Free;

               // Determine Fi, Di - detection and false positive rate of the current classifier
               AdjustBOnValidationSet(cascade[fNumCascade], fProps.minDetectRate[Min(Length(fProps.minDetectRate) - 1, fNumCascade)], F, D);

               inc(ni, featureIncrement);
               inc(numRounds);
          end;

          // reduce the data set and train the next stage only with false detected examples!
          for counter := DataSet.Count - 1 downto 0 do
              if cascade[fNumCascade].Classify(DataSet[counter]) = DataSet[counter].ClassVal then
                 DataSet.Delete(counter);

          inc(fNumCascade);
     end;

     SetLength(cascade, fNumCascade);

     DoProgress(100);
     Result := TCascadeBoostClassifier.Create(cascade, -1);
end;

procedure TCascadeBoostLearner.OnWeakClProgress(Sender: TObject;
  progress: integer);
var aProgress : integer;
begin
     aProgress := Min(99, (fNumCascade*100 + progress) div fProps.maxNumCascade );
     if aProgress <> fLastProgress then
     begin
          DoProgress( aProgress );
          fLastProgress := aProgress;
     end;
end;

procedure TCascadeBoostLearner.SetProperties(const Props: TCascadeBoostProps);
begin
     fProps := Props;
end;

destructor TCascadeBoostLearner.Destroy;
begin
     fRndEng.Free;

     inherited;
end;

{ TCascadeBoostClassifier }

class function TCascadeBoostClassifier.ClassIdentifier: String;
begin
     Result := 'BoostCascade';
end;

function TCascadeBoostClassifier.Classify(Example: TCustomExample;
  var confidence: double): integer;
var i : integer;
begin
     confidence := 0;
     Result := fRefClassVal;

     // ################################################
     // #### go through all cascade elements and classfy the example
     // -> if one creates a negative result (fRefClassVal) then we can break the loop
     i := 0;
     while i < Length(fCascade) do
     begin
          Result := fCascade[i].Classify(example, confidence);
          if Result = fRefClassVal then
             break;

          inc(i);
     end;
end;

constructor TCascadeBoostClassifier.Create(cascade: TCustomBoostArr; RefClassVal : integer);
begin
     fCascade := cascade;
     fRefClassVal := RefClassVal;

     inherited Create;
end;

procedure TCascadeBoostClassifier.DefineProps;
var i : Integer;
begin
     inherited;

     AddIntProperty(cBoostCascadeRefClassVal, fRefClassVal);
     BeginList(cBoostCascadeCascade, Length(fCascade));
     for i := 0 to Length(fCascade) - 1 do
         AddObject(fCascade[i]);
     EndList;
end;

function TCascadeBoostClassifier.PropTypeOfName(const Name: string): TPropType;
begin
     if CompareText(Name, cBoostCascadeRefClassVal) = 0
     then
         Result := ptInteger
     else if CompareText(Name, cBoostCascadeCascade) = 0
     then
         Result := ptObject
     else
         Result := inherited PropTypeOfName(Name);
end;


destructor TCascadeBoostClassifier.Destroy;
var i : integer;
begin
     for I := 0 to Length(fCascade) - 1 do
         fCascade[i].Free;
     fCascade := nil;

     inherited;
end;

procedure TCascadeBoostClassifier.OnLoadBeginList(const Name: String;
  count: integer);
begin
     fActIdx := -1;
     if CompareText(Name, cBoostCascadeCascade) = 0 then
     begin
          SetLength(fCascade, count);
          fActIdx := 0;
     end
     else
         inherited;
end;

procedure TCascadeBoostClassifier.OnLoadEndList;
begin
     if fActIdx >= 0
     then
         fActIdx := -1
     else
         inherited;
end;

procedure TCascadeBoostClassifier.OnLoadIntProperty(const Name: String;
  Value: integer);
begin
     if CompareText(Name, cBoostCascadeRefClassVal) = 0
     then
         fRefClassVal := Value
     else
         inherited;
end;

function TCascadeBoostClassifier.OnLoadObject(
  Obj: TBaseMathPersistence): boolean;
begin
     if fActIdx >= 0 then
     begin
          Result := True;
          fCascade[fActIdx] := Obj as TCustomBoostingClassifier;
          inc(fActIdx);
     end
     else
         Result := inherited OnLoadObject(Obj);
end;

initialization
  RegisterMathIO(TCascadeBoostClassifier);


end.
