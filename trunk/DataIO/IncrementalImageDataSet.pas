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

unit IncrementalImageDataSet;

// #############################################################
// #### Incremental data set loading for directories containing images
// #############################################################

interface

uses SysUtils, Classes, BaseIncrementalLearner, Types, Graphics, ImageMatrixConv;

// Loading of images from directcries
// -> first loads all filenames into memory
// -> then performs the incrementa classifier creation
type
  TIncrementalImageExampleList = class(TCustomIncrementalLearnerExampleList)
  private
    fBasePath : string;
    fFiles : TStringList;
    fImgWidth : integer;
    fImgHeight : integer;
    fImgResize: TOnImgResizeEvent;
    fConvType : TImageConvType;
    fNumClasses : integer;

    procedure InitFileList;
  protected
    function GetCount : integer; override;
  public
    property OnImgResizeEvent : TOnImgResizeEvent read fImgResize write fImgResize;

    procedure LoadExamples; override;

    constructor Create(const ImgPath : string; const aConvType : TImageConvType; const aInitPercentage : double; aStrategy : TIncrementalLearnStrategy);
    destructor Destroy; override;
  end;

implementation

uses Matrix, math, BaseMatrixExamples;

{ TIncrementalImageExampleList }

constructor TIncrementalImageExampleList.Create(const ImgPath: string; const aConvType : TImageConvType;
  const aInitPercentage: double; aStrategy: TIncrementalLearnStrategy);
begin
     inherited Create(aInitPercentage, aStrategy);

     fBasePath := IncludeTrailingPathDelimiter(Imgpath);
     fFiles := TStringList.Create;
     fConvType := aConvType;

     InitFileList;
end;

destructor TIncrementalImageExampleList.Destroy;
begin
     fFiles.Free;

     inherited;
end;

function TIncrementalImageExampleList.GetCount: integer;
begin
     Result := fFiles.Count;
end;

procedure TIncrementalImageExampleList.InitFileList;
var dirFiles : TSearchRec;
    registeredExtensions : TStringList;
    nonRegExtensions : TStringList;
procedure EnumFilesInDir(const dir : string);
var imgFiles : TSearchRec;
    ext : string;
    pict : TPicture;
begin
     // note it would be easier if we would have access to the function Graphics.GetFileFormats
     // function. Se I have to build it myself
     if FindFirst(fBasePath + dir + '*.*', faArchive, imgFiles) = 0 then
     begin
          repeat
                ext := ExtractFileExt(imgFiles.Name);

                if nonRegExtensions.IndexOf(ext) >= 0
                then
                    continue
                else if registeredExtensions.IndexOf(ext) >= 0
                then
                    fFiles.AddObject(dir + imgFiles.Name, TObject(fNumClasses))
                else
                begin
                     // test load the file -> if successfull then it's good
                     try
                        pict := TPicture.Create;
                        try
                           pict.LoadFromFile(fBasePath + dir + imgFiles.Name);

                           if fImgWidth = -1 then
                           begin
                                fImgWidth := pict.Width;
                                fImgHeight := pict.Height;
                           end;

                           registeredExtensions.Add(ExtractFileExt(imgFiles.Name));
                           fFiles.AddObject(dir + imgFiles.Name, TObject(fNumClasses))
                        finally
                               pict.Free;
                        end;
                     except
                           nonRegExtensions.Add(ExtractFileExt(imgFiles.Name));
                     end;
                end;
          until FindNext(imgFiles) <> 0;
     end;

     FindClose(imgFiles);
end;
begin
     // ###############################################################
     // #### First enumerate all files which can be read using the Delphi internal reading routines
     fFiles.Clear;
     
     registeredExtensions := TStringList.Create;
     nonRegExtensions := TStringList.Create;
     try
        fImgWidth := -1;
        fImgHeight := -1;
        fNumClasses := 0;

        if FindFirst(fBasePath + '*.*', faDirectory, dirFiles) = 0 then
        begin
             repeat
                   if (dirFiles.Name = '.') or (dirFiles.Name = '..') or ((dirFiles.Attr and faDirectory) = 0) then
                      continue;

                   EnumFilesInDir(IncludeTrailingPathDelimiter(dirFiles.Name));
                   inc(fNumClasses);
             until FindNext(dirFiles) <> 0;
        end;

        FindClose(dirFiles);
     finally
            nonRegExtensions.Free;
            registeredExtensions.Free;
     end;
end;

procedure TIncrementalImageExampleList.LoadExamples;
var imgData : TDoubleMatrix;
    classCnt : TIntegerDynArray;
    i : integer;
    j : integer;
    actIdx : integer;
    bmp : TBitmap;
    conv : TMatrixImageConverter;
    imgIdx : integer;
    mtxList : TMatrixLearnerExampleList;
    weights : TDoubleDynArray;
    classVals : TIntegerDynArray;
    unusedIdx : TIntegerDynArray;
    example : TMatrixLearnerExample;
    idx1, idx2 : integer;
    classImg : TDoubleMatrixDynArr;
    actClass : integer;
    numElem : integer;
    pict : TPicture;
begin
     SetLength(unusedIdx, fFiles.Count);
     for i := 0 to Length(unusedIdx) - 1 do
         unusedIdx[i] := i;

     conv := TMatrixImageConverter.Create(fConvType, False, False, fImgWidth, fImgHeight);
     try
        if Assigned(fOnLoadInitComplete) then
        begin
             // #########################################################
             // #### First initialize the classifier with a base dataset
             SetLength(classCnt, fNumClasses);
             for i := 0 to fFiles.Count - 1 do
                 inc(classCnt[Integer(fFiles.Objects[i])]);

             imgData := TDoubleMatrix.Create(Round(fFiles.Count*InitPercentage),
                        TMatrixImageConverter.MatrixHeightFromDim(fImgWidth, fImgHeight, fConvType));

             SetLength(weights, imgData.Width);
             SetLength(classVals, imgData.Width);

             // get from each class the same number of elements
             bmp := TBitmap.Create;
             try
                actIdx := 0;
                imgIdx := 0;
                for i := 0 to fNumClasses - 1 do
                begin
                     for j := 0 to Round(classCnt[i]*InitPercentage) - 1 do
                     begin
                          if imgIdx >= Round(fFiles.Count*InitPercentage) then
                             break;

                          pict := TPicture.Create;
                          try
                             pict.LoadFromFile(fBasePath + fFiles[actIdx]);
                             bmp.SetSize(pict.Width, pict.Height);
                             bmp.Canvas.Draw(0, 0, pict.Graphic);

                             if (bmp.Width <> fImgWidth) or (bmp.Height <> fImgHeight) and Assigned(fImgResize) then
                                fImgResize(Self, bmp, fImgWidth, fImgHeight);

                             if (bmp.Width <> fImgWidth) or (bmp.Height <> fImgHeight) then
                                raise Exception.Create('Error image properties do not match the expected ones');
                          finally
                                 pict.Free;
                          end;

                          unusedIdx[actIdx] := -1;
                          inc(actIdx);

                          imgData.SetSubMatrix(imgIdx, 0, 1, imgData.Height);
                          conv.ImageToMatrix(imgData, bmp);
                          classVals[imgIdx] := i;
                          weights[imgIdx] := 1;
                          inc(imgIdx);
                     end;

                     inc(actIdx, classCnt[i] - Round(classCnt[i]*InitPercentage));
                end;

                imgData.UseFullMatrix;
             finally
                    bmp.Free;
             end;

             imgData.SetSubMatrix(0, 0, imgIdx, imgData.Height);
             SetLength(classVals, imgIdx);
             weights := nil;

             if Assigned(fOnLoadInitComplete) then
             begin
                  mtxList := TMatrixLearnerExampleList.Create(imgData, classVals);
                  try
                     fOnLoadInitComplete(self, mtxList, weights);
                  finally
                         mtxList.Free;
                  end;
             end;

             // shrink unused list
             actIdx := 0;
             for i := 0 to Length(unusedIdx) - 1 do
             begin
                  if unusedIdx[i] = -1 then
                     continue;

                  unusedIdx[actIdx] := unusedIdx[i];
                  inc(actIdx); 
             end;

             SetLength(unusedIdx, actIdx);
        end;

        // #########################################################
        // #### Load the rest -> incrementaly
        if fLoadStrategy = lsRandom then
        begin
             for i := 0 to 2*Length(unusedIdx) - 1 do
             begin
                  idx1 := Random(Length(unusedIdx));
                  idx2 := Random(Length(unusedIdx));

                  actIdx := unusedIdx[idx1];
                  unusedIdx[idx1] := unusedIdx[idx2];
                  unusedIdx[idx2] := actIdx;
             end;
        end;
        if Assigned(fOnLoadExample) and (fLoadStrategy in [lsRandom, lsOneByOne]) then
        begin
             imgData := TDoubleMatrix.Create(1, TMatrixImageConverter.MatrixHeightFromDim(fImgWidth, fImgHeight, fConvType));
             bmp := TBitmap.Create;
             try
                for i := 0 to Length(unusedIdx) - 1 do
                begin
                     pict := TPicture.Create;
                     try
                        pict.LoadFromFile(fBasePath + fFiles[unusedIdx[i]]);
                        bmp.SetSize(pict.Width, pict.Height);
                        bmp.Canvas.Draw(0, 0, pict.Graphic);

                        if (bmp.Width <> fImgWidth) or (bmp.Height <> fImgHeight) and Assigned(fImgResize) then
                           fImgResize(Self, bmp, fImgWidth, fImgHeight);

                        if (bmp.Width <> fImgWidth) or (bmp.Height <> fImgHeight) then
                           raise Exception.Create('Error image properties do not match the expected ones');
                     finally
                            pict.Free;
                     end;

                     conv.ImageToMatrix(imgData, bmp);
                     example := TMatrixLearnerExample.Create(imgData, 0, Integer(fFiles.Objects[unusedIdx[i]]), False);
                     try
                        fOnLoadExample(Self, example);
                     finally
                            example.Free;
                     end;
                end;
             finally
                    imgData.Free;
                    bmp.Free;
             end;
        end;

        if Assigned(fOnLoadClass) and (fLoadStrategy = lsClassWise) and (Length(unusedIdx) > 0) then
        begin
             // get maximum size
             SetLength(classImg, 10);
             bmp := TBitmap.Create;
             try
                actClass := Integer(fFiles.Objects[unusedIdx[0]]);
                numElem := 0;
                for i := 0 to Length(unusedIdx) - 1 do
                begin
                     bmp.LoadFromFile(fBasePath + fFiles[unusedIdx[i]]);

                     if (bmp.Width <> fImgWidth) or (bmp.Height <> fImgHeight) then
                        if Assigned(fImgResize) then
                           fImgResize(Self, bmp, fImgWidth, fImgHeight);

                     if (bmp.Width <> fImgWidth) or (bmp.Height <> fImgHeight) then
                        raise Exception.Create('Error unexpected image size');

                     if (Integer(fFiles.Objects[unusedIdx[i]]) <> actClass) or (i = Length(unusedIdx) - 1) then
                     begin
                          imgData := TDoubleMatrix.Create(numElem, TMatrixImageConverter.MatrixHeightFromDim(fImgWidth, fImgHeight, fConvType));
                          SetLength(classVals, numElem);
                          SetLength(weights, numElem);
                          for j := 0 to numElem - 1 do
                          begin
                               imgData.SetColumn(j, classImg[j]);
                               FreeAndNil(classImg[j]);
                               classVals[j] := Integer(fFiles.Objects[unusedIdx[i - 1]]);
                               weights[j] := 1;
                          end;

                          mtxList := TMatrixLearnerExampleList.Create(imgData, classVals, True);
                          try
                             fOnLoadClass(Self, mtxList, Weights);
                          finally
                                 mtxList.Free;
                          end;

                          actClass := Integer(fFiles.Objects[unusedIdx[i]]);
                          numElem := 0;
                     end;

                     if numElem >= Length(classImg) then
                        SetLength(classImg, Min(2*Length(classImg), Length(classImg) + 100));
                     classImg[numElem] := conv.ImageToMatrix(bmp);
                     inc(numElem);
                end;
             finally
                    bmp.Free;
             end;
        end;
     finally
            conv.Free;
     end;
end;

end.
