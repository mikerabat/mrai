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

uses SysUtils, BaseClassifier, BaseMathPersistence, CustomBooster, RandomEng;

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
  TLearnerCreate = procedure(Sender : TObject; cl : TCustomBoostLearner) of Object;

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
    minFalsePosRate : Array of double;
    minDetectRate : Array of double;
    NumFeatures : Array of integer;
  end;

type
 TCascadeBoostLearner = class(TCustomLearner)
 private
   fProps : TCascadeBoostProps;
   fValidationSet : TCustomLearnerExampleList;
   fLearnerCreate : TLearnerCreate;
   fRndEng : TRandomGenerator;

   procedure AdjustBOnValidationSet(cl : TCustomClassifier; const desiredD : double; var F, D : double);
   procedure CreateValidationSet;
 protected
   function DoUnweightedLearn : TCustomClassifier; override;
 public
   property OnLearnerCreate : TLearnerCreate read fLearnerCreate write fLearnerCreate;
   procedure SetProperties(const Props : TCascadeBoostProps);

   constructor Create;
   destructor Destroy; override;
 end;

implementation

uses SimpleDecisionStump, AdaBoost, contnrs, math, types, MathUtilFunc;

const cBoostCascadeRefClassVal = 'boostCascadeRefCl';
      cBoostCascadeCascade = 'boostCascadeArr';

{ TCascadeBoostLearner }

type
  THackBoostCl = class(TCustomBoostingClassifier);

procedure TCascadeBoostLearner.AdjustBOnValidationSet(cl: TCustomClassifier; const desiredD : double; var F, D: double);
var counter : integer;
    classVal : integer;
    conf : TDoubleDynArray;
    idx : integer;
begin
     F := 0;
     D := 0;
     SetLength(conf, fValidationSet.Count);

     for counter := 0 to fValidationSet.Count - 1 do
     begin
          classVal := cl.Classify(fValidationSet.Example[counter], conf[counter]);

          conf[counter] := classVal*conf[counter];
          if classVal = fValidationSet.Example[counter].ClassVal
          then
              D := D + 1
          else
              F := F + 1;
     end;

     // set B such that the desired detection rate is met
     idx := Min(Length(conf) - 2, Round((1 - desiredD)*Length(conf)));

     if D < idx then
     begin
          // sort the list to ease the detection rate search
          QuickSort(conf, sizeof(double), Length(conf), DoubleSortFunc);

          THackBoostCl(cl).fB := (conf[idx] + conf[idx + 1])/2;

          // reevaluate validation set -> Get the real rates
          for counter := 0 to fValidationSet.Count - 1 do
          begin
               classVal := cl.Classify(fValidationSet.Example[counter]);

               if classVal = fValidationSet.Example[counter].ClassVal
               then
                   D := D + 1
               else
                   F := F + 1;
          end;
     end;

     // we have found a valid classifier with adjusted B
     // -> adjust the weights according to Lienhardt et al such that the overall
     // rate is increased
     // todo!

     D := D/fValidationSet.Count;
     F := F/fValidationSet.Count;
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
var clIdx : TIntegerDynArray;
    counter: Integer;
    numExmpl : integer;
    lastClassVal : integer;
    validationCnt : integer;
    obj : TCustomLearnerExample;
begin
     SetLength(clidx, DataSet.Count);

     validationCnt := Max(1, Round(fProps.validationPerc*DataSet.Count) div 2);

     fValidationSet := TCustomLearnerExampleList(DataSet.ClassType).Create;
     fValidationSet.OwnsObjects := DataSet.OwnsObjects;

     numExmpl := 0;
     lastClassVal := -1000000;
     for counter := 0 to DataSet.Count - 1 do
     begin
          if (DataSet[counter].ClassVal = lastClassVal) or (lastClassVal = -1000000) then
          begin
               lastClassVal := DataSet[counter].ClassVal;
               clIdx[numExmpl] := counter;
               inc(numExmpl);
          end;
     end;

     // randomized extraction
     while fValidationSet.Count < validationCnt do
     begin
          repeat
                counter := fRndEng.RandInt(numExmpl);
          until clIdx[counter] <> -1;

          obj := DataSet[counter];

          fValidationSet.Add(obj);
          DataSet[counter] := nil;

          clIdx[counter] := -1;
     end;
     DataSet.Pack;

     for counter := 0 to DataSet.Count - 1 do
     begin
          if DataSet[counter].ClassVal <> lastClassVal then
          begin
               lastClassVal := DataSet[counter].ClassVal;
               break;
          end;
     end;

     numExmpl := 0;
     for counter := 0 to DataSet.Count - 1 do
     begin
          if (DataSet[counter].ClassVal = lastClassVal) or (lastClassVal = -1000000) then
          begin
               clIdx[numExmpl] := counter;
               inc(numExmpl);
          end;
     end;

     // randomized extraction
     while fValidationSet.Count < validationCnt do
     begin
          repeat
                counter := fRndEng.RandInt(numExmpl);
          until clIdx[counter] <> -1;

          obj := DataSet[counter];

          fValidationSet.Add(obj);
          DataSet[counter] := nil;

          clIdx[counter] := -1;
     end;
     DataSet.Pack;
end;

function TCascadeBoostLearner.DoUnweightedLearn: TCustomClassifier;
var cascade : TCustomBoostArr;
    F, D : double;  // actual False positive (F) and Detection (D) rate
    prevF : double;
    numRounds : integer;
    ni : integer;
    boostProps : TBoostProperties;
    minFalsePosStageRate : double;
    learner : TCustomBoostLearner;
    counter: Integer;
    featureIncrement : integer;
begin
     CreateValidationSet;

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

     SetLength(cascade, 100);

     F := 1.0;
     D := 1.0;
     numRounds := 0;

     while (F > fProps.OverallFalsePosRate) and ((fProps.maxNumCascade <= 0) or (numRounds < fProps.maxNumCascade)) do
     begin
          ni := 1;
          if numRounds < Length(fProps.NumFeatures) then
             ni := fProps.NumFeatures[numRounds];

          minFalsePosStageRate := fProps.minFalsePosRate[Min(Length(fProps.minFalsePosRate), numRounds)];
          prevF := F;

          if fProps.featureIncrease < 1 then
          begin
               featureIncrement := ni;
               if numRounds > 0 then
                  featureIncrement := Round(cascade[numRounds - 1].Classifiers.Count*fProps.featureIncrease);
          end
          else
          begin
               featureIncrement := ni;
               if (numRounds > 0) and (cascade[numRounds - 1].Classifiers.Count > fProps.featureIncrease) then
                  featureIncrement := Round(fProps.featureIncrease);
          end;

          // adjust the numbers of features until a specific false positive rate is achivied
          while (F > minFalsePosStageRate*prevF) or (dataSet.Count < 3) do
          begin
               boostProps.NumRounds := ni;
               boostProps.PruneToLowestError := fProps.PruneToLowestError;
               boostProps.InitClassSpecificWeights := fProps.InitClassSpecificWeights;

               boostProps.OwnsLearner := False;
               boostProps.WeakLearner := fProps.WeakLearner.Create;

               // #######################################################
               // #### learn the classifier
               learner := fProps.learnerClass.Create;
               try
                  // step to update learner properties!
                  if Assigned(fLearnerCreate) then
                     fLearnerCreate(Self, learner);

                  cascade[numRounds] := learner.Learn as TCustomBoostingClassifier;
               finally
                      learner.Free;
               end;
               boostProps.WeakLearner.Free;

               // Determine Fi, Di - detection and classification rate of the current classifier
               AdjustBOnValidationSet(cascade[numRounds], fProps.minDetectRate[Min(Length(fProps.minDetectRate) - 1, numRounds)], F, D);

               inc(ni, featureIncrement);
          end;

          // reduce the data set and train the next stage only with false detected examples!
          for counter := DataSet.Count - 1 downto 0 do
              if cascade[numRounds].Classify(DataSet[counter]) = DataSet[counter].ClassVal then
                 DataSet.Delete(counter);
     end;

     Result := TCascadeBoostClassifier.Create(cascade, -1);
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
          if Result <> fRefClassVal then
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
     end
     else
         Result := inherited OnLoadObject(Obj);
end;

end.
