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

unit BaseClassifier;

// #############################################################
// #### Defines an interface for classification algorithms used in the
// #### Boosting algorithms
// #############################################################

interface

uses SysUtils, Classes, Contnrs, Types, BaseMathPersistence;

type
  TOnLearnIteration = procedure(Sender : TObject; progress : integer) of Object;
  ECustomClassifierException = class(Exception);
  TIntIntArray = Array of TIntegerDynArray;

// #############################################################
// #### most basic element - can be used in an example
  TCustomFeatureList = class(TObject)
  protected
    fFeatureVecLen : integer;

    function GetFeature(index : integer) : double; virtual; abstract;
    procedure SetFeature(index : integer; value : double); virtual; abstract;
  public
    property FeatureVec[index : integer] : double read GetFeature write SetFeature; default;
    property FeatureVecLen : integer read fFeatureVecLen;
    procedure SetFeatureVec(const Feature : Array of Double); virtual; abstract;
  end;

// #############################################################
// #### Base class for all examples - just an abstraction to get a base class
type
  TCustomExample = class(TObject)
  private
    fFeatureVec : TCustomFeatureList;
    fOwnsFeature : boolean;
  public
    property FeatureVec : TCustomFeatureList read fFeatureVec;

    constructor Create(FeatureFec : TCustomFeatureList; ownsFeatureVec : boolean);
    destructor Destroy; override;
  end;

type
  TCustomLearnerExample = class(TCustomExample)
  private
    fClassVal : integer;
  public
    property ClassVal : integer read fClassVal write fClassVal;
  end;

// #############################################################
// #### List of examples - used in the training phase
type
  TCustomExampleList = class(TObjectList)
  private
    function GetExample(index : integer) : TCustomExample;
    procedure SetExample(index : integer; Value : TCustomExample);
  public
    property Example[index : integer] : TCustomExample read GetExample write SetExample;
    procedure Add(Exmpl : TCustomExample);
  end;

// #############################################################
// #### A dataset of weighted examples
type
  TCustomLearnerExampleList = class(TObjectList)
  private
    function GetExample(index : integer) : TCustomLearnerExample; virtual;
    procedure SetExample(index : integer; Value : TCustomLearnerExample);
  public
    property Example[index : integer] : TCustomLearnerExample read GetExample write SetExample; default;
    procedure Add(Exmpl : TCustomLearnerExample);

    function NumClasses(var classVals : TIntegerDynArray) : integer;
  end;

// #############################################################
// #### Base classifier class
type
  TCustomClassifier = class(TBaseMathPersistence)
  public
    function Classify(Example : TCustomExample; var confidence : double) : integer; overload; virtual; abstract;
    function Classify(Example : TCustomExample) : integer; overload; virtual;
  end;
  TCustomClassifierClass = class of TCustomClassifier;

// ######################################################
// #### Base progress class for learning a classification rule
type
  TCommonLearnerProps = class(TObject)
  private
    fOnLearnProgress : TOnLearnIteration;
  protected
    procedure DoProgress(progress : integer);
  public
    property OnLearnProgress : TOnLearnIteration read fOnLearnProgress write fOnLearnProgress;

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; virtual; abstract;
  end;

// ######################################################
// #### All classifiers from this type support weighted examples
// in their learning steps - these types of classifiers are
// from interest when learning the adaboost classifier.
type
  TCustomWeightedLearner = class(TCommonLearnerProps)
  private
    fDataSet : TCustomLearnerExampleList;
  protected
    property DataSet : TCustomLearnerExampleList read fDataSet;
    function DoLearn(const weights : Array of double) : TCustomClassifier; virtual; abstract;

    // returns the indices of an sorted array of features starting from the lowest
    function CalcSortIdx(featureIdx : integer) : TIntegerDynArray; overload;
    function CalcSortIdx(const dataSetIdx : TIntegerDynArray; featureIdx : integer) : TIntegerDynArray; overload;

    procedure IdxQuickSort(const Values : TDoubleDynArray; var Idx : TIntegerDynArray; L, R : integer);

    function IndexOfClasses(var Idx : TIntIntArray; var classes : TIntegerDynArray) : integer;
  public
    procedure Init(DataSet : TCustomLearnerExampleList); virtual;
    function Learn(const weights : Array of double) : TCustomClassifier; overload;
    function Learn : TCustomClassifier; overload;
  end;
  TCustomWeightedLearnerClass = class of TCustomWeightedLearner;

// ######################################################
// #### Base classifier which cannot handle weigthing in the example list
type
  TCustomLearner = class(TCustomWeightedLearner)
  protected
    function DoUnweightedLearn : TCustomClassifier; virtual; abstract;
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
  public

  end;
  TCustomLearnerClass = class of TCustomLearner;

implementation

{ TCustomExampleList }

procedure TCustomExampleList.Add(Exmpl: TCustomExample);
begin
     inherited Add(Exmpl);
end;

function TCustomExampleList.GetExample(index: integer): TCustomExample;
begin
     assert((Items[index] is TCustomExample), 'Item is not a feature');
     Result := TCustomExample(Items[index]);
end;

procedure TCustomExampleList.SetExample(index: integer; Value: TCustomExample);
begin
     Items[index] := value;
end;

{ TCommonClassifierProps }

procedure TCommonLearnerProps.DoProgress(progress: integer);
begin
     if Assigned(fOnLearnProgress) then
        fOnLearnProgress(Self, progress);
end;

{ TCustomLearnerExampleList }

procedure TCustomLearnerExampleList.Add(Exmpl: TCustomLearnerExample);
begin
     inherited Add(Exmpl);
end;

function TCustomLearnerExampleList.GetExample(
  index: integer): TCustomLearnerExample;
begin
     assert((Items[index] is TCustomLearnerExample), 'Item is not a feature');
     Result := TCustomLearnerExample(Items[index]);
end;

function TCustomLearnerExampleList.NumClasses(
  var classVals: TIntegerDynArray): integer;
var counter, clCnt : integer;
begin
     SetLength(classVals, 2);

     // determine number of classes in the dataset
     Result := 0;
     for counter := 0 to Count - 1 do
     begin
          classVals[Result] := Example[counter].ClassVal;
          inc(Result);

          for clCnt := 0 to Result - 2 do
          begin
               if classVals[clCnt] = classVals[Result - 1] then
               begin
                    dec(Result);
                    break;
               end;
          end;

          if Result = Length(classVals) then
             SetLength(classVals, 2*Length(classVals));
     end;

     SetLength(classVals, Result);
end;

procedure TCustomLearnerExampleList.SetExample(index: integer;
  Value: TCustomLearnerExample);
begin
     Items[index] := Value;
end;

{ TCustomWeightedLearner }

procedure TCustomWeightedLearner.IdxQuickSort(const Values : TDoubleDynArray; var Idx : TIntegerDynArray; L, R : integer);
var I, J: Integer;
    T: integer;
    P : double;
begin
     // indexed quick sort implementation of for double values
     repeat
           I := L;
           J := R;
           P := values[Idx[(L + r) shr 1]];
           repeat
                 while values[Idx[i]] < P do
                       Inc(I);
                 while values[Idx[j]] > P do
                       Dec(J);
                 if I <= J then
                 begin
                      T := Idx[I];
                      Idx[I] := Idx[J];
                      Idx[J] := T;

                      Inc(I);
                      Dec(J);
                 end;
           until I > J;

           if L < J then
              IdxQuickSort(Values, Idx, L, J);
           L := I;
     until I >= R;
end;


function TCustomWeightedLearner.CalcSortIdx(featureIdx: integer): TIntegerDynArray;
var values : TDoubleDynArray;
    j : integer;
begin
     assert(Assigned(DataSet) and (DataSet.Count > 0), 'Error no data set assigned');
     assert(featureIdx < DataSet[0].FeatureVec.FeatureVecLen, 'Feature Index out of bounds');

     SetLength(Result, DataSet.Count);
     SetLength(values, DataSet.Count);
     for j := 0 to DataSet.Count - 1 do
     begin
          Result[j] := j;
          values[j] := DataSet[j].FeatureVec[featureIdx];
     end;

     IdxQuickSort(Values, Result, 0, DataSet.Count - 1);
end;

function TCustomWeightedLearner.CalcSortIdx(const dataSetIdx: TIntegerDynArray;
  featureIdx: integer): TIntegerDynArray;
var values : TDoubleDynArray;
    j : integer;
begin
     assert(Assigned(DataSet) and (DataSet.Count >= Length(dataSetIdx)), 'Error no data set assigned');
     assert(featureIdx < DataSet[0].FeatureVec.FeatureVecLen, 'Feature Index out of bounds');

     SetLength(Result, Length(dataSetIdx));
     SetLength(values, Length(dataSetIdx));
     for j := 0 to Length(dataSetIdx) - 1 do
     begin
          Result[j] := j;
          values[j] := DataSet[dataSetIdx[j]].FeatureVec[featureIdx];
     end;

     IdxQuickSort(values, Result, 0, Length(dataSetIdx) - 1);
end;

function TCustomWeightedLearner.IndexOfClasses(var Idx: TIntIntArray;
  var classes: TIntegerDynArray): integer;
var counter, clsCnt : integer;
    found : boolean;
    actClass : integer;
    i: Integer;
    numItems : TIntegerDynArray;
begin
     Result := 0;
     SetLength(classes, 10);
     SetLength(Idx, 10);
     SetLength(numItems, 10);

     // #########################################################
     // #### store example indexes - and the count in the first index
     for counter := 0 to DataSet.Count - 1 do
     begin
          found := False;
          actClass := DataSet[counter].ClassVal;

          for clsCnt := 0 to Result - 1 do
          begin
               if actClass = classes[clsCnt] then
               begin
                    found := True;
                    inc(numItems[clsCnt]);

                    if (Length(Idx[clsCnt]) <= numItems[clscnt]) then
                       SetLength(Idx[clsCnt], 20 + Length(Idx[clsCnt]));

                    Idx[clsCnt][numItems[clsCnt] - 1] := counter;
                    break;
               end;
          end;

          if not Found then
          begin
               if Length(Idx) - 1 <= Result then
               begin
                    SetLength(idx, Length(idx)*2);
                    SetLength(classes, Length(classes)*2);
                    SetLength(numItems, Length(classes)*2);

                    numItems[Result] := 0;
               end;

               SetLength(idx[Result], 10);
               idx[Result][0] := counter;

               classes[Result] := actClass;
               inc(numItems[Result]);
               inc(Result);
          end;
     end;

     SetLength(idx, Result);
     for i := 0 to Result - 1 do
         SetLength(idx[i], numItems[i]);

     SetLength(classes, Result);
end;

procedure TCustomWeightedLearner.Init(DataSet: TCustomLearnerExampleList);
begin
     fDataSet := DataSet;
end;

function TCustomWeightedLearner.Learn(const weights: array of double): TCustomClassifier;
begin
     assert(Assigned(fDataSet), 'Error, call init before learn');

     if High(weights) <> fDataSet.Count - 1 then
        raise Exception.Create('Number of weights differs from the number of examples');

     Result := DoLearn(weights);
end;

function TCustomWeightedLearner.Learn: TCustomClassifier;
var weights : TDoubleDynArray;
    i : integer;
begin
     assert(Assigned(fDataSet), 'Error, call init before learn');
     
     // learn the classifier without weighting -> create equal weights for each example
     SetLength(weights, fDataSet.Count);

     for i := 0 to Length(weights) - 1 do
         weights[i] := 1/Length(weights);

     Result := DoLearn(Weights);
end;

{ TCustomLearner }

function TCustomLearner.DoLearn(
  const weights: array of double): TCustomClassifier;
begin
     Result := DoUnweightedLearn;
end;

{ TCustomExample }

constructor TCustomExample.Create(FeatureFec: TCustomFeatureList; ownsFeatureVec : boolean);
begin
     fFeatureVec := FeatureFec;
     fOwnsFeature := ownsFeatureVec;
end;

destructor TCustomExample.Destroy;
begin
     if fOwnsFeature then
        fFeatureVec.Free;
        
     inherited;
end;

{ TCustomClassifier }

function TCustomClassifier.Classify(Example: TCustomExample): integer;
var conf : double;
begin
     Result := Classify(Example, conf);
end;

end.
