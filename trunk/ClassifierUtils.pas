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

unit ClassifierUtils;

// #####################################################
// #### Classification utilities mostly used to create special datasets
// #####################################################

interface

uses SysUtils, Classes, BaseClassifier;

function CreateBalancedDataSet(LearningSet : TCustomLearnerExampleList) : TCustomLearnerExampleList;
function CreateRandomDataSet(LearningSet : TCustomLearnerExampleList; Percentage : integer) : TCustomLearnerExampleList;
function CreateRandomizedBalancedDataSet(LearningSet : TCustomLearnerExampleList; Percentage : integer) : TCustomLearnerExampleList;

implementation

uses Math;

function InternalRandomDataSet(LearningSet : TCustomLearnerExampleList; StartIdx, EndIdx : integer; numElements : integer) : TCustomLearnerExampleList;
var idx : Array of integer;
    i : Integer;
    index1, index2 : integer;
    len : integer;
    tmp : integer;
begin
     // ensure that no double entries exists
     SetLength(idx, EndIdx - StartIdx + 1);
     for i := StartIdx to EndIdx do
         idx[i - StartIdx] := i;

     len := Length(idx);
     // permute the array twice as much as it's length
     for i := 0 to 2*Length(idx) - 1 do
     begin
          index1 := Random(len);
          index2 := Random(len);

          tmp := idx[index1];
          idx[index1] := idx[index2];
          idx[index2] := tmp;
     end;

     // now create the resulting array
     Result := LearningSet.ClassType.Create as TCustomLearnerExampleList;
     Result.OwnsObjects := False;
     Result.Capacity := len;
     for i := 0 to numElements - 1 do
         Result.Add(LearningSet[idx[i]]);
end;

function CreateRandomDataSet(LearningSet : TCustomLearnerExampleList; Percentage : integer) : TCustomLearnerExampleList;
var numElements : integer;
begin
     Result := nil;
     numElements := Min(LearningSet.Count, (Percentage*LearningSet.Count) div 100);
     if numElements < 0 then
        exit;

     Result := InternalRandomDataSet(LearningSet, 0, LearningSet.Count - 1, numElements);
end;

function ClassSort(Item1, Item2 : Pointer) : integer;
begin
     Result := TCustomLearnerExample(Item1).ClassVal - TCustomLearnerExample(Item2).ClassVal;
end;

function CreateBalancedDataSet(LearningSet : TCustomLearnerExampleList) : TCustomLearnerExampleList;
var classes : Array of integer;
    numClasses : integer;
    i : integer;
    copyList : TCustomLearnerExampleList;
    minNumElem : integer;
    actNumElem : integer;
    actClass : integer;
begin
     Result := nil;
     if LearningSet.Count = 0 then
        exit;

     // we only want to store references to the examples in the new data set
     copyList := LearningSet.ClassType.Create as TCustomLearnerExampleList;
     try
        copyList.OwnsObjects := False;
        copyList.Capacity := LearningSet.Count;

        // first check out the number of classes and the number of elements belonging to these classes
        for i := 0 to LearningSet.Count - 1 do
            copyList.Add(LearningSet[i]);
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
        Result := LearningSet.ClassType.Create as TCustomLearnerExampleList;
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

function CreateRandomizedBalancedDataSet(LearningSet : TCustomLearnerExampleList; Percentage : integer) : TCustomLearnerExampleList;
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
     if LearningSet.Count = 0 then
        exit;

     // we only want to store references to the examples in the new data set
     copyList := LearningSet.ClassType.Create as TCustomLearnerExampleList;
     try
        copyList.OwnsObjects := False;
        copyList.Capacity := LearningSet.Count;

        // first check out the number of classes and the number of elements belonging to these classes
        for i := 0 to LearningSet.Count - 1 do
            copyList.Add(LearningSet[i]);
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
        Result := LearningSet.ClassType.Create as TCustomLearnerExampleList;
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

end.
