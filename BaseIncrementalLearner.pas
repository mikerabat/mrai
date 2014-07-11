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

unit BaseIncrementalLearner;

// #############################################################
// #### base incremental versions of the learners
// #############################################################

interface

uses SysUtils, Classes, BaseClassifier, Types;

type
  TLoadExampleEvent = procedure(Sender : TObject; Example : TCustomLearnerExample; const weight : double = -1) of object;
  TLoadInitCompleteEvent = procedure(Sender : TObject; Examples : TCustomLearnerExampleList; const Weights : TDoubleDynArray) of object;
  TLoadClassExamplesEvent = TLoadInitCompleteEvent;

type
  TIncrementalLearnStrategy = (lsOneByOne, lsRandom, lsClassWise);

// #############################################################
// #### Base class for incremental data list generation
type
  TCustomIncrementalLearnerExampleList = class(TObject)
  protected
    fOnLoadExample : TLoadExampleEvent;
    fInitPercentage : double;
    fOnLoadInitComplete: TLoadInitCompleteEvent;
    fLoadStrategy: TIncrementalLearnStrategy;
    fOnLoadClass: TLoadClassExamplesEvent;

    property InitPercentage : double read FInitPercentage;
    property LoadStrategy : TIncrementalLearnStrategy read fLoadStrategy;

    function GetCount : integer; virtual; abstract;
  public
    property OnLoadExample : TLoadExampleEvent read fOnLoadExample write fOnLoadExample;
    property OnLoadClass : TLoadClassExamplesEvent read fOnLoadClass write fOnLoadClass;
    property OnLoadInitComplete : TLoadInitCompleteEvent read fOnLoadInitComplete write fOnLoadInitComplete;
    property Count : integer read GetCount;

    procedure LoadExamples; virtual; abstract;

    constructor Create(const aInitPercentage : double; aStrategy : TIncrementalLearnStrategy);
  end;

// #############################################################
// #### Base class for incremental learning - a learning method must implement
// Event handlers.
// The executed strategy is defined in the following:
// * After reading aInitPercentage of the whole data set the OnLoadInitComplete is executed
//   thus the classifier can init with a base dataset.
// * Depending on the Learn strategy either one by one example is read or
//   a whole class is loaded into memory and learnd.
type
  TCustomIncrementalWeightedLearner = class(TCommonLearnerProps)
  private
    fDataSet : TCustomIncrementalLearnerExampleList;
  protected
    function IndexOfClasses(DataSet : TCustomLearnerExampleList; var Idx : TIntIntArray; var classes : TIntegerDynArray) : integer; overload;
    function IndexOfClasses(const DataClassLabels : TIntegerDynArray; numExamples : integer; var idx : TIntIntArray; var classes : TIntegerDynArray) : integer; overload;
    
    function GetClassifier : TCustomClassifier; virtual; abstract;

    property DataSet : TCustomIncrementalLearnerExampleList read fDataSet;
    property Classifier : TCustomClassifier read GetClassifier;

    procedure OnLoadExample(Sender : TObject; Example : TCustomLearnerExample; const weight : double = -1); virtual; abstract;
    procedure OnLoadClass(Sender : TObject; Examples : TCustomLearnerExampleList; const Weights : TDoubleDynArray); virtual; abstract;
    procedure OnLoadInitComplete(Sender : TObject; Examples : TCustomLearnerExampleList; const Weights : TDoubleDynArray); virtual; abstract;
  public
    function Learn(const weights : Array of double) : TCustomClassifier; overload; virtual;
    function Learn : TCustomClassifier; overload;
    procedure Init(DataSet : TCustomIncrementalLearnerExampleList); virtual;
  end;
  TCustomIncrementalWeightedLearnerClass = class of TCustomIncrementalWeightedLearner;


implementation

uses Math;

{ TCustomIncrementalWeightedLearner }

function TCustomIncrementalWeightedLearner.Learn(const weights: array of double): TCustomClassifier;
begin
     // note the examples are loaded one by one -> the OnLoadExample procedure must create
     // the classifier which is returned
     DataSet.LoadExamples;
     Result := Classifier;
end;

function TCustomIncrementalWeightedLearner.IndexOfClasses(DataSet : TCustomLearnerExampleList;
 var Idx: TIntIntArray; var classes: TIntegerDynArray): integer;
var counter, clsCnt : integer;
    found : boolean;
    actClass : integer;
begin
     Result := 0;
     SetLength(classes, 10);
     SetLength(Idx, 10);

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
                    inc(Idx[clsCnt][0]);

                    if (Length(Idx[clsCnt]) <= Idx[clsCnt][0]) then
                       SetLength(Idx[clsCnt], 20 + Length(Idx[clsCnt]));

                    Idx[clsCnt][Idx[clsCnt][0]] := counter;
                    break;
               end;
          end;

          if not Found then
          begin
               if Length(Idx) - 1 <= Result then
               begin
                    SetLength(idx, Length(idx)*2);
                    SetLength(classes, Length(classes)*2);
               end;

               SetLength(idx[Result], 10);
               idx[Result][0] := 1;
               idx[Result][1] := counter;

               classes[Result] := actClass;

               inc(Result);
          end;
     end;

     SetLength(classes, Result);
end;

function TCustomIncrementalWeightedLearner.IndexOfClasses(
  const DataClassLabels : TIntegerDynArray; numExamples : integer; var idx : TIntIntArray;
  var classes : TIntegerDynArray) : integer;
var counter, clsCnt : integer;
    found : boolean;
    actClass : integer;
begin
     Result := 0;
     SetLength(Idx, 10);
     SetLength(classes, 10);

     // #########################################################
     // #### store example indexes - and the count in the first index
     for counter := 0 to numExamples - 1 do
     begin
          found := False;
          actClass := DataClassLabels[counter];

          for clsCnt := 0 to Result - 1 do
          begin
               if actClass = classes[clsCnt] then
               begin
                    found := True;
                    inc(Idx[clsCnt][0]);

                    if (Length(Idx[clsCnt]) <= Idx[clsCnt][0]) then
                       SetLength(Idx[clsCnt], 20 + Length(Idx[clsCnt]));

                    Idx[clsCnt][Idx[clsCnt][0]] := counter;
                    break;
               end;
          end;

          if not Found then
          begin
               if Length(Idx) - 1 <= Result then
               begin
                    SetLength(idx, Length(idx)*2);
                    SetLength(classes, Length(classes)*2);
               end;

               SetLength(idx[Result], 10);
               idx[Result][0] := 1;
               idx[Result][1] := counter;

               classes[Result] := actClass;

               inc(Result);
          end;
     end;

     SetLength(classes, Result);
end;

procedure TCustomIncrementalWeightedLearner.Init(
  DataSet: TCustomIncrementalLearnerExampleList);
begin
     fDataSet := DataSet;

     fDataSet.OnLoadExample := OnLoadExample;
     fDataSet.OnLoadInitComplete := OnLoadInitComplete;
     fDataSet.OnLoadClass := OnLoadClass;
end;

function TCustomIncrementalWeightedLearner.Learn: TCustomClassifier;
var weights : Array of double;
    i: Integer;
    weight : double;
begin
     SetLength(weights, DataSet.Count);
     weight := 1/DataSet.Count;
     for i := 0 to Length(weights) - 1 do
         weights[i] := weight;

     Result := Learn(weights);
end;

{ TCustomIncrementalLearnerExampleList }

constructor TCustomIncrementalLearnerExampleList.Create(
  const aInitPercentage: double; aStrategy : TIncrementalLearnStrategy);
begin
     fInitPercentage := Max(0, Min(1, aInitPercentage));
     fLoadStrategy := aStrategy;

     inherited Create;
end;

end.
