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

unit SortedExampleLearner;

// #############################################################
// #### Base class of all (weighted) learners who need
// #### a sorted example list
// #############################################################

interface

uses SysUtils, Classes, BaseClassifier, Types;

const cStumpInitProgres = 30;

type
  TIntIntDynArray = Array of TIntegerDynArray;

type
  TCustomSortedExampleLearner = class(TCustomWeightedLearner)
  private
    fSortIdxs : TIntIntDynArray;
    procedure InitFunc(DataSet : TCustomLearnerExampleList; const values : TDoubleDynArray; column : integer);
    procedure ExampleSort(const values : TDoubleDynArray; Column : integer; L, R: Integer);
  protected
    property SortIdx : TIntIntDynArray read fSortIdxs;
  public
    procedure Init(DataSet : TCustomLearnerExampleList); override;
  end;

implementation

uses Windows, SyncObjs;

var NumIterThreads : integer = 2;

type
  TSortThrList = class;
  TSortThread = class(TThread)
  private
    fRef : TSortThrList;
  protected
    procedure Execute; override;
  public
    constructor Create(ref : TSortThrList);
  end;

  TSortThrList = class(TObject)
  protected
    fSortThreads : Array of TSortThread;
    fLearner : TCustomSortedExampleLearner;
    fDataSet : TCustomLearnerExampleList;
    fActIdx : integer;
    fCS : TCriticalSection;
    fFeatureVecLen : integer;
    fCnt : integer;
  public
    function GetNext : integer;
    procedure AsyncSort;

    function Finished(var progress : integer) : boolean;

    constructor Create(el : TCustomSortedExampleLearner; dataSet : TCustomLearnerExampleList);
    destructor Destroy; override;
  end;

procedure TCustomSortedExampleLearner.Init(DataSet : TCustomLearnerExampleList);
var i : integer;
    evalThreads : TSortThrList;
    values : TDoubleDynArray;
    progress : integer;
    oldPrg : integer;
begin
     inherited;

     assert(DataSet.Count > 0, 'Cannot learn empty dataset');
     assert(DataSet[0].FeatureVec.FeatureVecLen > 0, 'Cannot learn empty feature vectors');

     exit;

     // ###############################################
     // #### Create sorted index array for fast learning
     SetLength(fSortIdxs, DataSet.Count, DataSet[0].FeatureVec.FeatureVecLen);

     if NumIterThreads <= 1 then
     begin
          SetLength(values, DataSet.Count);
          // for each dimension sort the index array
          for i := 0 to DataSet[0].FeatureVec.FeatureVecLen - 1 do
          begin
               InitFunc(DataSet, values, i);

               DoProgress(Round(cStumpInitProgres*i/DataSet[0].FeatureVec.FeatureVecLen));
          end;
     end
     else
     begin
          evalThreads := TSortThrList.Create(self, DataSet);
          try
             // #################################################
             // #### threaded feature sorting -> spawn one thread for each avail cpu
             evalThreads.AsyncSort;

             oldPrg := 0;
             // #################################################
             // #### Wait until all sorting has finished
             while not evalThreads.Finished(progress) do
             begin
                  if oldPrg <> progress then
                  begin
                       DoProgress(Round(progress*100/cStumpInitProgres));
                       oldPrg := progress;
                  end;
             end;
          finally
                 evalThreads.Free;
          end;
     end;

     DoProgress(cStumpInitProgres);
end;

procedure TCustomSortedExampleLearner.InitFunc(
  DataSet: TCustomLearnerExampleList; const values: TDoubleDynArray;
  column: integer);
var j : integer;
begin
     for j := 0 to DataSet.Count - 1 do
     begin
          fSortIdxs[j][column] := j;
          values[j] := DataSet[j].FeatureVec[column];
     end;
     ExampleSort(values, column, 0, DataSet.Count - 1);
end;

procedure TCustomSortedExampleLearner.ExampleSort(const values : TDoubleDynArray; Column: integer; L, R: Integer);
var I, J: Integer;
    T: integer;
    P : double;
begin
     // indexed quick sort implementation of for double values
     repeat
           I := L;
           J := R;
           P := values[fSortIdxs[(L + r) shr 1][Column]];
           repeat
                 while values[fSortIdxs[i][Column]] < P do
                       Inc(I);
                 while values[fSortIdxs[j][Column]] > P do
                       Dec(J);
                 if I <= J then
                 begin
                      T := fSortIdxs[I][Column];
                      fSortIdxs[I][Column] := fSortIdxs[J][Column];
                      fSortIdxs[J][Column] := T;

                      Inc(I);
                      Dec(J);
                 end;
           until I > J;

           if L < J then
              ExampleSort(values, Column, L, J);
           L := I;
     until I >= R;
end;

{ TSortThread }

constructor TSortThread.Create(ref: TSortThrList);
begin
     fref := ref;

     FreeOnTerminate := False;
     inherited Create(True);
end;

procedure TSortThread.Execute;
var actIdx : integer;
    values : TDoubleDynArray;
begin
     SetLength(values, fRef.fDataSet.Count);

     while not Terminated do
     begin
          // #################################################
          // #### Fetch then next index to sort
          actIdx := fRef.GetNext;

          if actIdx < 0 then
             break;

          // the object knows what to dow with the values
          fRef.fLearner.InitFunc(fRef.fDataSet, values, actIdx);
     end;
end;

// #################################################
// #### Thread variables initialization
// #################################################

var sysInfo : TSystemInfo;

{ TSortThrList }

procedure TSortThrList.AsyncSort;
var i : Integer;
begin
     fCnt := fDataSet.Count;
     assert(fCnt > 0, 'Error empty data set');
     fFeatureVecLen := fDataSet[0].FeatureVec.FeatureVecLen;
     fActIdx := -1;

     for i := 0 to Length(fSortThreads) - 1 do
         fSortThreads[i].Start;
end;

constructor TSortThrList.Create(el: TCustomSortedExampleLearner;
  dataSet: TCustomLearnerExampleList);
var i : integer;
begin
     fLearner := el;
     fDataSet := dataSet;
     fCS := TCriticalSection.Create;

     SetLength(fSortThreads, NumIterThreads);

     for i := 0 to Length(fSortThreads) - 1 do
         fSortThreads[i] := TSortThread.Create(self);
end;

destructor TSortThrList.Destroy;
var i : Integer;
begin
     fCS.Free;

     for i := 0 to Length(fSortThreads) - 1 do
         fSortThreads[i].Free;

     inherited;
end;

function TSortThrList.Finished(var progress: integer): boolean;
var hdls : Array of THandle;
    i : Integer;
begin
     SetLength(hdls, Length(fSortThreads));
     for i := 0 to Length(fSortThreads) - 1 do
         hdls[i] := fSortThreads[i].Handle;

     Result := WaitForMultipleObjects(Length(fSortThreads), @hdls[0], True, 100) <> WAIT_TIMEOUT;

     progress := fActIdx*100 div fDataSet.Count;
end;

function TSortThrList.GetNext: integer;
begin
     // todo: could be made with the interlockedinc and such functions
     Result := -1;
     fCS.Enter;
     try
        if fActIdx = fFeatureVecLen - 1 then
           exit;

        inc(fActIdx);

        Result := fActIdx;
     finally
            fCS.Leave;
     end;
end;

initialization
  GetSystemInfo(SysInfo);
  NumIterThreads := SysInfo.dwNumberOfProcessors;

end.
