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

uses SysUtils, Classes, Contnrs, Types, BaseMathPersistence, RandomEng;

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

    constructor Create(FeatureVec : TCustomFeatureList; ownsFeatureVec : boolean);
    destructor Destroy; override;
  end;

type
  TCustomLearnerExample = class(TCustomExample)
  private
    fClassVal : integer;
  public
    function Clone : TCustomLearnerExample; virtual;

    property ClassVal : integer read fClassVal write fClassVal;
  end;
  TCustomLearnerExampleClass = class of TCustomLearnerExample;

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
    fRandomAlg : TRandomAlgorithm;
    fRandom : TRandomGenerator;

    function InternalRandomDataSet(LearningSet : TCustomLearnerExampleList; StartIdx, EndIdx : integer; numElements : integer) : TCustomLearnerExampleList;
    procedure SetRandomAlg(const Value: TRandomAlgorithm);

    function GetExample(index : integer) : TCustomLearnerExample; 
    procedure SetExample(index : integer; Value : TCustomLearnerExample);
  public
    procedure CreateTrainAndValidationSet(validationDataSetPerc : integer; out trainSet, validationSet : TCustomLearnerExampleList);
    function CreateBalancedDataSet : TCustomLearnerExampleList;
    function CreateRandomDataSet(Percentage : integer) : TCustomLearnerExampleList;
    function CreateRandomizedBalancedDataSet(Percentage : integer) : TCustomLearnerExampleList;

    // clone without examples
    function CloneBase : TCustomLearnerExampleList; virtual;

    function Shuffle : TIntegerDynArray;  // randomizes all the examples
    function Rand : TRandomGenerator;

    property Example[index : integer] : TCustomLearnerExample read GetExample write SetExample; default;
    procedure Add(Exmpl : TCustomLearnerExample);

    function NumClasses(var classVals : TIntegerDynArray) : integer;

    property RandomAlg : TRandomAlgorithm read fRandomAlg write SetRandomAlg;
    constructor Create;
    destructor Destroy; override;
  end;
  TCustomLearnerExampleListClass = class of TCustomLearnerExampleList;

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

    function CountSortIdx(const dataSetIdx: TIntegerDynArray; featureIdx: integer): TIntegerDynArray; 

    procedure IdxCountSort(const Values: TIntegerDynArray; var Idx: TIntegerDynArray; Min, Max: integer); // by reference
    procedure IdxQuickSort(const Values : TDoubleDynArray; var Idx : TIntegerDynArray; L, R : integer);

    function IndexOfClasses(var Idx : TIntIntArray; var classes : TIntegerDynArray) : integer;
    function Classes : TIntegerDynArray;
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
  private
    fOrigDataSet : TCustomLearnerExampleList;

    // creates a new example list and adds already existing examples from the given list:
    // weighting is achieved by duplicating
    // items. e.g. if count=3 and weights are 0.66, 0.17, 0.17, then the
    // result is count=5, example[0] x 3, example[1] x 1, example[2] x 1
    // algorithm is:
    // take max weigth and divide by 100: -> use that as minimum allowed weight and
    // discard all examples lower than that weight.
    // take the remaining lowest weight -> this example is put one time into the resulting
    // array.
    procedure BuildWeightedList(const weights : Array of double);
  protected
    function DoUnweightedLearn : TCustomClassifier; virtual; abstract;
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
  end;
  TCustomLearnerClass = class of TCustomLearner;

implementation

uses Math, BaseMatrixExamples;

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
     //assert((Items[index] is TCustomLearnerExample), 'Item is not a feature');
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

constructor TCustomLearnerExampleList.Create;
begin
     inherited Create(True);

     fRandomAlg := raMersenneTwister;
end;

function TCustomLearnerExampleList.InternalRandomDataSet(LearningSet : TCustomLearnerExampleList; StartIdx, EndIdx : integer; numElements : integer) : TCustomLearnerExampleList;
var idx : Array of integer;
    i : Integer;
    index : integer;
    len : integer;
    tmp : integer;
begin
     // ensure that no double entries exists
     SetLength(idx, EndIdx - StartIdx + 1);
     for i := StartIdx to EndIdx do
         idx[i - StartIdx] := i;

     len := Length(idx);

     // Fisher yates shuffle:
     for i := Length(idx) - 1 downto 1 do
     begin
          index := Rand.RandInt(i + 1);

          tmp := idx[index];
          idx[index] := idx[i];
          idx[i] := tmp;
     end;

     // now create the resulting array
     Result := TCustomLearnerExampleList.Create; // LearningSet.ClassType.Create as TCustomLearnerExampleList;
     Result.OwnsObjects := False;
     Result.Capacity := len;
     for i := 0 to numElements - 1 do
         Result.Add(LearningSet[idx[i]]);
end;

function TCustomLearnerExampleList.CreateRandomDataSet(Percentage : integer) : TCustomLearnerExampleList;
var numElements : integer;
begin
     Result := nil;
     numElements := Min(Count, (Percentage*Count) div 100);
     if numElements < 0 then
        exit;

     Result := InternalRandomDataSet(self, 0, Count - 1, numElements);
end;

function ClassSort(Item1, Item2 : Pointer) : integer;
begin
     Result := TCustomLearnerExample(Item1).ClassVal - TCustomLearnerExample(Item2).ClassVal;
end;

function TCustomLearnerExampleList.CreateBalancedDataSet : TCustomLearnerExampleList;
var classes : Array of integer;
    numClasses : integer;
    i : integer;
    copyList : TCustomLearnerExampleList;
    minNumElem : integer;
    actNumElem : integer;
    actClass : integer;
begin
     Result := nil;
     if Count = 0 then
        exit;

     // we only want to store references to the examples in the new data set
     copyList := ClassType.Create as TCustomLearnerExampleList;
     try
        copyList.OwnsObjects := False;
        copyList.Capacity := Count;

        // first check out the number of classes and the number of elements belonging to these classes
        for i := 0 to Count - 1 do
            copyList.Add(Example[i]);
        copyList.Sort(ClassSort);

        SetLength(classes, 10);
        numClasses := 1;
        classes[0] := 1;

        for i := 1 to copyList.Count - 1 do
        begin
             if copyList[i].ClassVal <> copyList[i - 1].ClassVal then
             begin
                  inc(NumClasses);

                  if NumClasses >= Length(classes) then
                     SetLength(classes, Min(2*Length(classes), Length(classes) + 1000));
             end;

             inc(classes[numClasses - 1]);
        end;

        // search for the class with the lowest number of elements
        minNumElem := classes[0];
        for i := 1 to numClasses - 1 do
            minNumElem := Min(minNumElem, classes[i]);

        // create the resulting list:
        Result := TCustomLearnerExampleList.Create; //ClassType.Create as TCustomLearnerExampleList;
        Result.OwnsObjects := False;
        Result.Capacity := minNumElem*numClasses;

        actNumElem := 0;
        actClass := 0;
        for i := 0 to copyList.Count - 1 do
        begin
             if actNumElem = classes[actClass] then
             begin
                  inc(actClass);
                  actNumElem := 0;
             end;

             if actNumElem < minNumElem then
                Result.Add(copyList[i]);

             inc(actNumElem);
        end;
     finally
            copyList.Free;
     end;
end;

function TCustomLearnerExampleList.CreateRandomizedBalancedDataSet(Percentage : integer) : TCustomLearnerExampleList;
var classes : Array of integer;
    numClasses : integer;
    i, j : integer;
    copyList : TCustomLearnerExampleList;
    minNumElem : integer;
    actNumElem : integer;
    actClass : integer;
    list : TCustomLearnerExampleList;
begin
     Result := nil;
     if Count = 0 then
        exit;

     // we only want to store references to the examples in the new data set
     copyList := TCustomLearnerExampleList.Create; //ClassType.Create as TCustomLearnerExampleList;
     try
        copyList.OwnsObjects := False;
        copyList.Capacity := Count;

        // first check out the number of classes and the number of elements belonging to these classes
        for i := 0 to Count - 1 do
            copyList.Add(Example[i]);
        copyList.Sort(ClassSort);

        SetLength(classes, 10);
        numClasses := 1;
        classes[0] := 1;

        for i := 1 to copyList.Count - 1 do
        begin
             if copyList[i].ClassVal <> copyList[i - 1].ClassVal then
             begin
                  inc(NumClasses);

                  if NumClasses >= Length(classes) then
                     SetLength(classes, Min(2*Length(classes), Length(classes) + 1000));
             end;

             inc(classes[numClasses - 1]);
        end;

        // search for the class with the lowest number of elements
        minNumElem := classes[0];
        for i := 1 to numClasses - 1 do
            minNumElem := Min(minNumElem, classes[i]);

        minNumElem := (minNumElem*Max(0, Min(100, Percentage))) div 100;

        // create the resulting list:
        Result := TCustomLearnerExampleList.Create; //ClassType.Create as TCustomLearnerExampleList;
        Result.OwnsObjects := False;
        Result.Capacity := minNumElem*numClasses;

        actNumElem := 0;
        actClass := 0;
        for i := 0 to numClasses - 1 do
        begin
             // this line ensures that consecutive calls to this routine does not result in the same resulting dataset
             list := InternalRandomDataSet(copyList, actNumElem, actNumElem + Classes[actClass] - 1, minNumElem);
             try
                for j := 0 to list.Count - 1 do
                    Result.Add(list[j]);
             finally
                    list.Free;
             end;
             inc(actNumElem, Classes[actClass]);
             inc(actClass);
        end;
     finally
            copyList.Free;
     end;
end;

function TCustomLearnerExampleList.Rand: TRandomGenerator;
begin
     if not Assigned(fRandom) then
     begin
          fRandom := TRandomGenerator.Create;
          fRandom.RandMethod := RandomAlg;
          fRandom.Init(0);
     end;

     Result := fRandom;
end;


procedure TCustomLearnerExampleList.SetRandomAlg(const Value: TRandomAlgorithm);
begin
     fRandomAlg := Value;

     if Assigned(fRandom) then
        FreeAndNil(fRandom);
end;

destructor TCustomLearnerExampleList.Destroy;
begin
     fRandom.Free;

     inherited;
end;

procedure TCustomLearnerExampleList.CreateTrainAndValidationSet(
  validationDataSetPerc : integer; out trainSet,
  validationSet: TCustomLearnerExampleList);
var counter : integer;
    numValidationElem : integer;
begin
     assert((validationDataSetPerc <= 100) and (validationDataSetPerc >= 0), 'Percentage needs to be between 0 and 100');
     trainSet := CreateRandomDataSet(100);

     validationSet := self.ClassType.Create as TCustomLearnerExampleList;

     // special case: 0 or 100 - just clone the train set: validation set == trainSet
     if (validationDataSetPerc = 0) or (validationDataSetPerc = 100) then
     begin
          validationSet.Capacity := trainSet.Count;
          for counter := 0 to trainset.Count - 1 do
              validationSet.Add(trainset[counter]);
     end
     else
     begin
          numValidationElem := Max(1, Floor(validationDataSetPerc/100*trainSet.Count));
          for counter := 0 to numValidationElem - 1 do
          begin
               validationSet.Add(trainSet[trainSet.Count - 1 - counter]);
               trainSet[trainSet.Count - 1 - counter] := nil;
          end;
          // delete last elements
          trainSet.Pack;
     end;
end;

function TCustomLearnerExampleList.Shuffle : TIntegerDynArray;
var i : integer;
    index : integer;
    tmp : integer;
begin
     SetLength(Result, Count);
     for i := 0 to Count - 1 do
         Result[i] := i;

     // Fisher yates shuffle:
     for i := Count - 1 downto 1 do
     begin
          index := Rand.RandInt(i + 1);

          tmp := Result[i];
          Result[i] := Result[index];
          Result[index] := tmp;
     end;
end;

function TCustomLearnerExampleList.CloneBase: TCustomLearnerExampleList;
begin
     Result := TCustomLearnerExampleListClass(Self.ClassType).Create;
     Result.fRandomAlg := fRandomAlg;
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
     if DataSet is TMatrixLearnerExampleList then
     begin
          for j := 0 to Length(dataSetIdx) - 1 do
          begin
               Result[j] := j;
               values[j] := TMatrixLearnerExampleList(DataSet).Matrix[dataSetIdx[j], featureIdx];
          end;
     end
     else
     begin
          for j := 0 to Length(dataSetIdx) - 1 do
          begin
               Result[j] := j;
               values[j] := DataSet[dataSetIdx[j]].FeatureVec[featureIdx];
          end;
     end;

     IdxQuickSort(values, Result, 0, Length(dataSetIdx) - 1);
end;

procedure TCustomWeightedLearner.IdxCountSort(const Values : TIntegerDynArray; var Idx : TIntegerDynArray; Min,Max : integer);
Var count: Array of integer;
    I, n : integer;
begin
     SetLength(count, max - min + 1);
     n := Length(Values);

     for I := 0 to (max - min) do count[I] := 0;
         for I := 0 to (n - 1) do
             count[Values[I] - min] := count[Values[I] - min] + 1;

     // compute the total
     Count[0] := Count[0]-1;  //make sure it starts at zero
     for I := 1 to (max - min) do 
         count[I] := Count[I-1]+Count[i];


     for I := n-1 downto 0 do 
     begin
          Idx[Count[Values[i]-min]] := I;
          Dec(Count[Values[i]-min]);
     end;
end;

function TCustomWeightedLearner.CountSortIdx(const dataSetIdx: TIntegerDynArray;
   featureIdx : integer): TIntegerDynArray;
var values : TIntegerDynArray;
    j : integer;
    minVal, maxVal : integer;
begin
     assert(Assigned(DataSet) and (DataSet.Count >= Length(dataSetIdx)), 'Error no data set assigned');
     assert(featureIdx < DataSet[0].FeatureVec.FeatureVecLen, 'Feature Index out of bounds');

     SetLength(Result, Length(dataSetIdx));
     SetLength(values, Length(dataSetIdx));

     minVal := MaxInt;
     maxval := -MaxInt;
     if DataSet is TMatrixLearnerExampleList then
     begin
          for j := 0 to Length(dataSetIdx) - 1 do
          begin
               values[j] := Round(TMatrixLearnerExampleList(DataSet).Matrix[dataSetIdx[j], featureIdx]);
               minVal := min(minVal, values[j]);
               maxVal := max(maxVal, values[j]);
          end;
     end
     else
     begin
          for j := 0 to Length(dataSetIdx) - 1 do
          begin
               Values[j] := Round(DataSet[DataSetIdx[j]].FeatureVec[FeatureIdx]);
               minVal := min(minVal, values[j]);
               maxVal := max(maxVal, values[j]);
          end;
     end;
     
     IdxCountSort(values, Result, MinVal, MaxVal);
end;

function TCustomWeightedLearner.Classes: TIntegerDynArray;
var counter, clsCnt : integer;
    found : boolean;
    actClass : integer;
    numClasses : integer;
begin
     SetLength(Result, 10);
     numClasses := 0;

     // #########################################################
     // #### store class indicess in the output array
     for counter := 0 to DataSet.Count - 1 do
     begin
          found := False;
          actClass := DataSet[counter].ClassVal;

          for clsCnt := 0 to numClasses - 1 do
          begin
               if actClass = Result[clsCnt] then
               begin
                    found := True;
                    break;
               end;
          end;

          if not Found then
          begin
               if Length(Result) - 1 <= numClasses then
                  SetLength(Result, Length(Result)*2);

               Result[numClasses] := actClass;
               inc(numClasses);
          end;
     end;

     SetLength(Result, numClasses);
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
     // build a new list and use it like the original one
     BuildWeightedList(weights);
     try
        Result := DoUnweightedLearn;
     finally
            // cleanup the intermediate dataset
            if Assigned(fOrigDataSet) then
            begin
                 fDataSet.Free;
                 fDataSet := fOrigDataSet;
                 fOrigDataSet := nil;
            end;
     end;
end;

procedure TCustomLearner.BuildWeightedList(const weights: array of double);
var maxWeight : double;
    counter: Integer;
    minAllowedWeight : double;
    minWeight : double;
    numExmpl : integer;
    i : integer;
    exmpl : TCustomLearnerExample;
begin
     // search for the maximum:
     if length(weights) = 0 then
        exit;

     maxWeight := weights[0];
     for counter := 1 to Length(weights) - 1 do
         maxWeight := max(maxWeight, weights[counter]);

     minAllowedWeight := maxWeight/100;
     // find minimum
     minWeight := maxWeight;
     for counter := 0 to Length(weights) - 1 do
     begin
          if (weights[counter] < minWeight) and (weights[counter] >= minAllowedWeight) then
             minWeight := weights[counter];
     end;

     // check if evenly distributed - if so do nothing
     if minWeight = maxWeight then
        exit;

     fOrigDataSet := fDataSet;
     fDataSet := fOrigDataSet.CloneBase;
     fDataSet.fRandomAlg := fOrigDataSet.fRandomAlg;

     // build resulting example list and multiple add the examples according to the weighting
     for counter := 0 to Length(weights) - 1 do
     begin
          // remove all examples lower than the given weight
          if weights[counter] < minWeight
          then
              continue
          else
          begin
               numExmpl := ceil(weights[counter]/minWeight);
               // clone and add examples
               for i := 0 to numExmpl - 1 do
               begin
                    exmpl := TCustomLearnerExample(fOrigDataSet.GetExample(counter).Clone);
                    fDataSet.Add(exmpl);
               end;
          end;
     end;
end;

{ TCustomExample }

constructor TCustomExample.Create(FeatureVec: TCustomFeatureList; ownsFeatureVec : boolean);
begin
     fFeatureVec := FeatureVec;
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

{ TCustomLearnerExample }

function TCustomLearnerExample.Clone: TCustomLearnerExample;
begin
     Result := TCustomLearnerExampleClass(self.ClassType).Create(fFeatureVec, False);
     Result.fClassVal := fClassVal;
end;

end.
