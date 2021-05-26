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

unit SimpleDecisionStump;

// #############################################################
// #### Implementation of a simple decission stump for the
// #### boosting algorithms
// #############################################################

interface

uses SysUtils, Classes, Types, BaseClassifier, BaseMathPersistence;

// ####################################################
// #### simple decision stump classifier.
// can be seen as a decision tree with only a root node.
// The classifier consists of a threshold t and the index of the
// dimension on which the classification shall take place.
// as confidence the margin between the threshold and the example is returned.
// todo: eventually use some kind of gaussian distribution for the confidence calculation.
type
  TDecisionStump = class(TCustomClassifier)
  private
    fDimension : LongInt;
    fThreshold : double;
    fClass1, fClass2 : LongInt;
    fConfMult : double;
    fConfBias : double;  // the learning error
  public
    property Dimension : integer read fDimension;
    property Threshold : double read fThreshold;

    function Classify(Example : TCustomExample; var confidence : double) : integer; overload; override;
    function Classify(Example : TCustomExample) : integer; overload; override;

    // loading and saving procedures
    procedure OnLoadDoubleProperty(const Name : String; const Value : double); override;
    procedure OnLoadIntProperty(const Name : String; Value : integer); override;

    procedure DefineProps; override;
    function PropTypeOfName(const Name : string) : TPropType; override;

    constructor Create(Dim : integer; Thresh : double; c1, c2 : integer; const confMult, confBias : double);
  end;

// #####################################################
// #### Decission stump learner. The learning method first sorts all examples according
// their feature vector values. This reduces the computation time for iterative learning steps with
// weighted datasets.
// see: Viola Jones: Robust Real-Time Face Detection
type
  TDecisionStumpLearner = class(TCustomWeightedLearner)
  private
    fWeights : PDouble;
    fErrors : TDoubleDynArray;
    fThresh : TDoubleDynArray;
    fMinIdx : TIntegerDynArray;

    procedure MaintainSums(featureIdx: integer; const SortIdx : TIntegerDynArray; const Weights : Array of double;
                           var TotSumP, TotSumN : double; var SumNeg, SumPos : TDoubleDynArray; var c1, c2 : integer);
  protected
    procedure CalcOneFeature( featureIndex : integer; var sumNeg, sumPos : TDoubleDynArray );
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
  public
    procedure Init(DataSet : TCustomLearnerExampleList); override;

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
  end;

implementation

uses Math, Windows, SyncObjs, MatrixConst;

const cDimensionProp = 'dimension';
      cThresholdProp = 'threshold';
      cClass1Prop = 'class1';
      cClass2Prop = 'class2';
      cConfProp = 'conf';
      cConfBiasProp = 'trainerror';

// ######################################################
// #### Helper classes for a multithreaded version of the decission stump learner

type
  TErrorEvalThrList = class;
  TErrorEvalThread = class(TThread)
  private
    fRef : TErrorEvalThrList;
    fEvt : TSimpleEvent;
    fFinished : boolean;
  protected
    procedure Execute; override;
  public
    property Finished : boolean read fFinished;
    procedure SetEvent;
    constructor Create(ref : TErrorEvalThrList);
    destructor Destroy; override;
  end;

  TErrorEvalThrList = class(TObject)
  protected
    fEvalThreads : Array of TErrorEvalThread;
    fLearner : TDecisionStumpLearner;
    fActIdx : integer;
    fFeatureVecLen : integer;
    fCS : TCriticalSection;
    fEvt : TSimpleEvent;
  public
    function GetNext(var idx : integer) : boolean;
    procedure AsyncEval(el : TDecisionStumpLearner);

    function Finished(var progress : integer) : boolean;

    constructor Create;
    destructor Destroy; override;
  end;

var NumIterThreads : integer = 1;
    LocThrList : TErrorEvalThrList = nil;
    LocLock : TCriticalSection = nil;


{ TErrorEvalThread }

function DecStumpThrList : TErrorEvalThrList;
begin
     if not Assigned(LocThrList) then
        LocThrList := TErrorEvalThrList.Create;

     Result := LocThrList;
end;

constructor TErrorEvalThread.Create(ref: TErrorEvalThrList);
begin
     fFinished := False;
     fRef := ref;
     fEvt := TSimpleEvent.Create(nil, True, False, '');

     inherited Create(False);
end;

destructor TErrorEvalThread.Destroy;
begin
     fEvt.Free;

     inherited;
end;

procedure TErrorEvalThread.Execute;
var featureIdx : integer;
    wr : TWaitResult;
    sumNeg, sumPos : TDoubleDynArray;
begin
     while not Terminated do
     begin
          wr := fEvt.WaitFor(5000);

          if wr <> wrSignaled then
             continue;

          if Terminated then
             break;

          fFinished := False;
             
          while fRef.GetNext(featureIdx) do
          begin
               // ###########################################
               // #### Maintain the list for one feature
               fRef.fLearner.CalcOneFeature(featureIdx, sumNeg, sumPos);
          end;

          fFinished := True;
          
          // reset to wait for next learning event
          fEvt.ResetEvent;
     end;
end;

procedure TErrorEvalThread.SetEvent;
begin
     fFinished := False;
     fEvt.SetEvent;
end;

{ TErrorEvalThrList }

procedure TErrorEvalThrList.AsyncEval(el : TDecisionStumpLearner);
var i: Integer;
begin
     fLearner := el;
     fFeatureVecLen := fLearner.DataSet[0].FeatureVec.FeatureVecLen;

     fActIdx := -1;
     for i := 0 to Min(fFeatureVecLen, NumIterThreads) - 1 do
         fEvalThreads[i].SetEvent;
end;

constructor TErrorEvalThrList.Create;
var i : integer;
begin
     fLearner := nil;
     fCS := TCriticalSection.Create;
     fActIdx := -1;
     fFeatureVecLen := 0;
     fEvt := TSimpleEvent.Create;

     SetLength(fEvalThreads, NumIterThreads);
     for i := 0 to NumIterThreads - 1 do
         fEvalThreads[i] := TErrorEvalThread.Create(self);

     inherited Create;
end;

destructor TErrorEvalThrList.Destroy;
var i : integer;
begin
     for i := 0 to Length(fEvalThreads) - 1 do
     begin
          fEvalThreads[i].Terminate;
          fEvalThreads[i].SetEvent;

          fEvalThreads[i].WaitFor;
          fEvalThreads[i].Free;
     end;

     fCS.Free;
     fEvt.Free;

     inherited;
end;

function TErrorEvalThrList.Finished(var progress: integer): boolean;
var i: Integer;
begin
     fEvt.WaitFor(20);

     fCS.Enter;
     try
        Result := fActIdx >= fFeatureVecLen;
        if Result then
        begin
             progress := 100;

             // ###########################################
             // #### We need to wait since it may be that 
             // threads are still evaluating
             for i := 0 to Min(fFeatureVecLen, NumIterThreads) - 1 do
                 Result := Result and fEvalThreads[i].Finished;
        end
        else
            progress := (100*fActIdx) div fFeatureVecLen;
     finally
            fCS.Leave;
     end;

     fEvt.ResetEvent;
end;

function TErrorEvalThrList.GetNext(var idx: integer): boolean;
begin
     // #################################################
     // #### Thread save incremting
     fCS.Enter;
     try
        inc(fActIdx);

        Result := fActIdx < fFeatureVecLen;
        idx := fActIdx;
     finally
            fCS.Leave;
     end;

     fEvt.SetEvent;
end;

// #####################################################
// #### Actual learner
{ TDecisionStumpLearner }

procedure TDecisionStumpLearner.CalcOneFeature(featureIndex: integer; var sumNeg, sumPos : TDoubleDynArray);
var TotSumP : double;
    TotSumN : double;
    i : integer;
    e : double;
    c1, c2 : integer;
    minIdx : integer;
    SortIdx : TIntegerDynArray;
begin
     c1 := DataSet[0].ClassVal;
     c2 := -MaxInt;
     
     // #################################################
     // #### do some precalculations to find the minimum made error
     // for that specific feature easily
     SortIdx := CalcSortIdx(featureIndex);
     MaintainSums(featureIndex, SortIdx, Slice(PConstDoubleArr(fWeights)^, DataSet.Count), 
                  TotSumP, TotSumN, SumNeg, SumPos, c1, c2);

     // calculate minimum error for this feature
     fErrors[featureIndex] := MaxDouble;
     minIdx := -1;
     for i := 0 to DataSet.Count - 1 do
     begin
          e := Min(SumPos[i] + (TotSumN - SumNeg[i]), SumNeg[i] + (TotSumP - SumPos[i]));

          if e < fErrors[featureIndex] then
          begin
               minIdx := i;
               fErrors[featureIndex] := e;
          end;
     end;

     // actual min index is the sorted feature index
     fMinIdx[featureIndex] := SortIdx[minIdx];

     assert(minIdx >= 0, 'error in error calculation');

     if minIdx < Length(sortIdx) - 1
     then
         fThresh[featureIndex] := (DataSet[SortIdx[minIdx]].FeatureVec[featureIndex] + DataSet[SortIdx[minIdx + 1]].FeatureVec[featureIndex]) / 2
     else
         fThresh[featureIndex] := DataSet[SortIdx[minIdx]].FeatureVec[featureIndex];
end;

class function TDecisionStumpLearner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := Classifier = TDecisionStump;
end;

procedure TDecisionStumpLearner.MaintainSums(featureIdx : integer; const SortIdx : TIntegerDynArray;
  const Weights : Array of double; var TotSumP, TotSumN : double; var SumNeg, SumPos : TDoubleDynArray;
  var c1, c2 : integer);
var i : integer;
begin
     if Length(sumNeg) <> DataSet.Count then
        SetLength(sumNeg, dataSet.Count);
     if Length(sumPos) <> DataSet.Count then
        SetLength(sumPos, dataSet.Count);
     // first calculate total sums
     TotSumP := 0;
     TotSumN := 0;
     SumNeg[0] := 0;
     SumPos[0] := 0;
     if DataSet[SortIdx[0]].ClassVal = c1 then
     begin
          TotSumP := weights[SortIdx[0]];
          SumPos[0] := weights[SortIdx[0]];
     end
     else
     begin
          TotSumN := weights[SortIdx[0]];
          SumNeg[0] := weights[SortIdx[0]];
     end;

     // maintain sums
     for i := 1 to DataSet.Count - 1 do
     begin
          if DataSet[SortIdx[i]].ClassVal = c1 then
          begin
               TotSumP := TotSumP + weights[SortIdx[i]];
               SumPos[i] := SumPos[i - 1] + weights[SortIdx[i]];
               SumNeg[i] := SumNeg[i - 1];
          end
          else
          begin
               c2 := DataSet[SortIdx[i]].ClassVal;
               TotSumN := TotSumN + weights[SortIdx[i]];
               SumPos[i] := SumPos[i - 1];
               SumNeg[i] := SumNeg[i - 1] + weights[SortIdx[i]];
          end;
     end;
end;

function TDecisionStumpLearner.DoLearn(
  const weights: array of double): TCustomClassifier;
var TotSumP : double;
    TotSumN : double;
    SumPos : TDoubleDynArray;
    SumNeg : TDoubleDynArray;
    i, j : integer;
    e : double;
    Dim : integer;
    c1, c2 : integer;
    confMult : double;
    diff : double;
    thresh : double;
    minError : double;
    numCorrectClassified : integer;
    progress : integer;
    SortIdx : TIntegerDynArray;
begin
     assert(DataSet.Count > 0, 'Error, cannot learn classifier from an empty dataset');

     fWeights := @weights[0];

     // ###########################################
     // #### Learn a decission stump according to the weights only
     // see: Viola Jones: Robust Real-Time Face Detection
     // the algorithm evaluates the weighted list in linear time
     SetLength(fErrors, DataSet[0].FeatureVec.FeatureVecLen);
     SetLength(fThresh, DataSet[0].FeatureVec.FeatureVecLen);
     SetLength(fMinIdx, DataSet[0].FeatureVec.FeatureVecLen);
     SetLength(SumPos, DataSet.Count);
     SetLength(SumNeg, DataSet.Count);

     c1 := DataSet[0].ClassVal;
     c2 := -MaxInt;

     // #####################################################
     // #### Multi threaded search for the optimal feature vector
     if NumIterThreads > 1 then
     begin
          LocLock.Enter;
          try
             DecStumpThrList.AsyncEval(self);

             sleep(10);
             while not DecStumpThrList.Finished(progress) do
                   DoProgress(progress);
          finally
                 LocLock.Leave;
          end;
     end
     else
     begin
          // evaluate all features in one loop
          for j := 0 to DataSet[0].FeatureVec.FeatureVecLen - 1 do
          begin
               CalcOneFeature(j, SumNeg, SumPos);
               
               DoProgress(round(100*j/DataSet[0].FeatureVec.FeatureVecLen));
          end;
     end;

     // ###################################################
     // #### Create decission stump classifier according to the calculated index and error
     // -> use feature as decission stump which has the lowest error rate:
     minError := fErrors[0];
     thresh := fThresh[0];
     dim := 0;
     for j := 1 to DataSet[0].FeatureVec.FeatureVecLen - 1 do
     begin
          if minError > fErrors[j] then
          begin
               dim := j;
               minError := fErrors[j];
               thresh := fThresh[j];
          end;
     end;

     // #####################################################
     // #### Calculate the factor for an arctan confidence calculation:
     // the point we search for is where the arctan function crosses 1 (at 1.5574)
     // this is the point where we define a confidence of 2/pi (63%).
     // the point is calculated according to the left and right training error.
     // where the distance reaches a training error of 63% we have our value.
     SortIdx := CalcSortIdx(dim);
     TotSumP := 0;
     TotSumN := 0;
     MaintainSums(dim, SortIdx, weights, TotSumP, TotSumN, SumNeg, SumPos, c1, c2);
     confMult := 0;
     for i := 0 to fMinIdx[dim] do
     begin
          e := Min(SumPos[i] + (TotSumN - SumNeg[i]), SumNeg[i] + (TotSumP - SumPos[i]));
          if e < 1 - 0.63 then
          begin
               diff := abs(DataSet[SortIdx[i]].FeatureVec[dim] - thresh);
               confMult := 1.5574/(diff + 0.001)/(1 - 0.37 + e);
               break;
          end;
     end;

     for i := DataSet.Count - 1 downto fMinIdx[dim] + 1 do
     begin
          e := Min(SumPos[i] + (TotSumN - SumNeg[i]), SumNeg[i] + (TotSumP - SumPos[i]));
          if e < 1 - 0.63 then
          begin
               diff := abs(DataSet[SortIdx[i]].FeatureVec[dim] - thresh);
               diff := 1.5574/(diff + 0.001)/(1 - 0.37 + e);

               confMult := (confMult + diff)/2;
               break;
          end;
     end;

     // ########################################################
     // #### check polarity
     numCorrectClassified := 0;

     for i := 0 to DataSet.Count - 1 do
     begin
          if ((DataSet[i].FeatureVec[dim] > thresh) and (DataSet[i].ClassVal = c1)) or
             ((DataSet[i].FeatureVec[dim] <= thresh) and (DataSet[i].ClassVal = c2))
          then
              inc(numCorrectClassified);
     end;

     DoProgress(100);

     // check polarity
     if numCorrectClassified >= DataSet.Count div 2
     then
         Result := TDecisionStump.Create(dim, thresh, c1, c2, confMult, 1 - minError)
     else
         Result := TDecisionStump.Create(dim, thresh, c2, c1, confMult, 1 - minError);
end;

procedure TDecisionStumpLearner.Init(DataSet: TCustomLearnerExampleList);
begin
     inherited;
end;

{ TDecisionStump }

function TDecisionStump.Classify(Example: TCustomExample;
  var confidence: double): integer;
var diff : double;
begin
     diff := Example.FeatureVec[fDimension] - fThreshold;
     Result := IfThen(diff > 0, fClass1, fClass2);

     // define the confidence as an arcus tangens function
     confidence := Min(1, Max(0, fConfBias + (1 - fConfBias)*1/1.5574*ArcTan(diff*fConfMult)));
end;

function TDecisionStump.Classify(Example: TCustomExample): integer;
var diff : double;
begin
     diff := Example.FeatureVec[fDimension] - fThreshold;
     Result := IfThen(diff > 0, fClass1, fClass2);
end;

constructor TDecisionStump.Create(Dim: integer; Thresh: double; c1, c2 : integer; const confMult, confBias : double);
begin
     inherited Create;

     fDimension := Dim;
     fThreshold := Thresh;
     fClass1 := c1;
     fClass2 := c2;
     fConfMult := confMult;
     fConfBias := confBias;
end;

procedure TDecisionStump.DefineProps;
begin
     AddIntProperty(cDimensionProp, fDimension);
     AddDoubleProperty(cThresholdProp, fThreshold);
     AddIntProperty(cClass1Prop, fClass1);
     AddIntProperty(cClass2Prop, fClass2);
     AddDoubleProperty(cConfProp, fConfMult);
     AddDoubleProperty(cConfBiasProp, fConfBias);
end;

function TDecisionStump.PropTypeOfName(const Name: string): TPropType;
begin
     if (CompareText(Name, cDimensionProp) = 0) or (CompareText(Name, cClass1Prop) = 0) or
        (CompareText(Name, cClass2Prop) = 0)
     then
         Result := ptInteger
     else if (CompareText(Name, cThresholdProp) = 0) or (CompareText(Name, cConfProp) = 0) or
             (CompareText(Name, cConfBiasProp) = 0)
     then
         Result := ptDouble
     else
         Result := inherited PropTypeOfName(Name);
end;


procedure TDecisionStump.OnLoadDoubleProperty(const Name: String;
  const Value: double);
begin
     if CompareText(name, cThresholdProp) = 0
     then
         fThreshold := Value
     else if CompareText(name, cConfProp) = 0
     then
         fConfMult := Value
     else if CompareText(name, cConfBiasProp) = 0
     then
         fConfBias := Value
     else
         inherited;
end;

procedure TDecisionStump.OnLoadIntProperty(const Name: String;
  Value: integer);
begin
     if CompareText(name, cDimensionProp) = 0
     then
         fDimension := Value
     else if CompareText(name, cClass1Prop) = 0
     then
         fClass1 := Value
     else if CompareText(name, cClass2Prop) = 0
     then
         fClass2 := Value
     else
         inherited;
end;

// #############################################################
// ##### Initialization

var SysInfo : TSystemInfo;

initialization
  GetSystemInfo(SysInfo);
  NumIterThreads := SysInfo.dwNumberOfProcessors;
  LocLock := TCriticalSection.Create;
  LocThrList := TErrorEvalThrList.Create;

  RegisterMathIO(TDecisionStump);

finalization
  LocLock.Free;
  LocThrList.Free;


end.
