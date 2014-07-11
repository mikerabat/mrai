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

unit ImageDataSet;

// #############################################################
// #### Data set loading for directories containing images
// #############################################################

interface

uses Windows, SysUtils, Classes, Graphics, BaseMatrixExamples, Matrix, Types,
     ImageMatrixConv;

// #############################################################
// #### Loading of a complete list at once
type
  TImageMatrixExampleList = class(TMatrixLearnerExampleList)
  private
    fImgHeight: integer;
    fImgWidth: integer;
    fImgResize: TOnImgResizeEvent;
    function LoadImages(const directory : string; var classVals : TIntegerDynArray; convType : TImageConvType) : TDoubleMatrix;
  public
    property OnImgResizeEvent : TOnImgResizeEvent read fImgResize write fImgResize;

    property ImgWidth : integer read fImgWidth;
    property ImgHeight : integer read fImgHeight;

    constructor Create(const dir : string; convType : TImageConvType; weightsFile : string = '');
  end;

implementation

uses Math;

{ TImageMatrixExampleList }

constructor TImageMatrixExampleList.Create(const dir: string; convType : TImageConvType;
  weightsFile: string);
var classVals : TIntegerDynArray;
    mtx : TDoubleMatrix;
begin
     mtx := LoadImages(dir, classVals, convType);
     assert(Assigned(mtx) and (mtx.Width = Length(classVals)), 'Load dimension error');

     inherited Create(mtx, classVals, True);
end;

function TImageMatrixExampleList.LoadImages(const directory : string; var classVals: TIntegerDynArray; convType : TImageConvType): TDoubleMatrix;
var actClass : integer;
    dirFiles : TSearchRec;
    numFiles : integer;
    filenames : TStringList;
    registeredExtensions : TStringList;
    nonRegExtensions : TStringList;
    imgIdx : integer;
    i : integer;
    pict : TPicture;
    bmp : TBitmap;
    converter : TMatrixImageConverter;
function EnumFilesInDir(const dir : string) : TStringList;
var imgFiles : TSearchRec;
    ext : string;
    pict : TPicture;
begin
     // note it would be easier if we would have access to the function Graphics.GetFileFormats
     // function. So I have to build it myself
     Result := TStringList.Create;
     
     if FindFirst(IncludeTrailingPathDelimiter(dir) + '*.*', faArchive, imgFiles) = 0 then
     begin
          repeat
                ext := ExtractFileExt(imgFiles.Name);

                if nonRegExtensions.IndexOf(ext) >= 0
                then
                    continue
                else if registeredExtensions.IndexOf(ext) >= 0
                then
                    Result.Add(IncludeTrailingPathDelimiter(dir) + imgFiles.Name)
                else
                begin
                     // test load the file -> if successfull then it's good
                     try
                        pict := TPicture.Create;
                        try
                           pict.LoadFromFile(IncludeTrailingPathDelimiter(dir) + imgFiles.Name);

                           if fImgWidth = -1 then
                           begin
                                fImgWidth := pict.Width;
                                fImgHeight := pict.Height;
                           end;

                           registeredExtensions.Add(ExtractFileExt(imgFiles.Name));
                           Result.Add(IncludeTrailingPathDelimiter(dir) + imgFiles.Name)
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
     registeredExtensions := TStringList.Create;
     nonRegExtensions := TStringList.Create;
     fImgWidth := -1;
     fImgHeight := -1;
     try
        numFiles := 0;
        if FindFirst(IncludeTrailingPathDelimiter(directory) + '*.*', faAnyFile, dirFiles) = 0 then
        begin
             repeat
                   if (dirFiles.Name = '.') or (dirFiles.Name = '..') or ((dirFiles.Attr and faDirectory) = 0) then
                      continue;

                   filenames := EnumFilesInDir(IncludeTrailingPathDelimiter(directory) + dirFiles.Name);
                   inc(numFiles, FileNames.Count);
                   filenames.Free;

             until FindNext(dirFiles) <> 0;
        end;

        FindClose(dirFiles);

        // ###############################################################
        // #### First enumerate all files
        SetLength(classVals, numFiles);
        converter := TMatrixImageConverter.Create(convType, False, False, fImgWidth, fImgHeight);
        try
           Result := TDoubleMatrix.Create(numFiles, converter.MatrixHeight);
           try
              actClass := 0;
              imgIdx := 0;

              if FindFirst(IncludeTrailingPathDelimiter(directory) + '*.*', faDirectory, dirFiles) = 0 then
              begin
                   repeat
                         if (dirFiles.Name = '.') or (dirFiles.Name = '..') or ((dirFiles.Attr and faDirectory) = 0) then
                            continue;

                         filenames := EnumFilesInDir(IncludeTrailingPathDelimiter(directory) + dirFiles.Name);

                         // #################################################################
                         // #### Create resulting matrix from the current image set
                         for i := 0 to filenames.Count - 1 do
                         begin
                              classVals[imgIdx] := actClass;
                              Result.SetSubMatrix(imgIdx, 0, 1, Result.Height);
                              pict := TPicture.Create;
                              try
                                 pict.LoadFromFile(filenames[i]);
                                 bmp := TBitmap.Create;
                                 try
                                    bmp.SetSize(pict.Width, pict.Height);
                                    bmp.Canvas.Draw(0, 0, pict.Graphic);

                                    if (bmp.Width <> fImgWidth) or (bmp.Height <> fImgHeight) and Assigned(fImgResize) then
                                       fImgResize(Self, bmp, fImgWidth, fImgHeight);

                                    if (bmp.Width <> fImgWidth) or (bmp.Height <> fImgHeight) then
                                       raise Exception.Create('Error image properties do not match the expected ones');

                                    converter.ImageToMatrix(Result, bmp);
                                 finally
                                        bmp.Free;
                                 end;
                              finally
                                     pict.Free;
                              end;
                              inc(imgIdx);
                         end;

                         filenames.Free;

                         inc(actClass);
                   until FindNext(dirFiles) <> 0;
              end;

              FindClose(dirFiles);

              Result.UseFullMatrix;
           except
                 Result.Free;

                 raise;
           end;
        finally
               converter.Free;
        end;
     finally
            registeredExtensions.Free;
            nonRegExtensions.Free;
     end;
end;

end.
