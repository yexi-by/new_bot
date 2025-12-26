from typing import Literal
from pydantic import BaseModel


class TextData(BaseModel):
    text: str


class AtData(BaseModel):
    qq: str | int


class ImageData(BaseModel):
    file: str
    url: str | None = None


class ReplyData(BaseModel):
    id: str | int


class FaceData(BaseModel):
    id: str | int


class FileData(BaseModel):
    file: str


class VideoData(BaseModel):
    file: str


class RecordData(BaseModel):
    file: str


class Text(BaseModel):
    """发送文本"""

    type: Literal["text"] = "text"
    data: TextData

    @classmethod
    def new(cls, text: str):
        return cls(data=TextData(text=text))


class At(BaseModel):
    """发送群艾特"""

    type: Literal["at"] = "at"
    data: AtData

    @classmethod
    def new(cls, qq: str):
        return cls(data=AtData(qq=qq))


class Image(BaseModel):
    """发送图片"""

    type: Literal["image"] = "image"
    data: ImageData

    @classmethod
    def new(cls, file: str):
        return cls(data=ImageData(file="base64://" + file))


class Reply(BaseModel):
    """发送回复"""

    type: Literal["reply"] = "reply"
    data: ReplyData

    @classmethod
    def new(cls, id: str | int):
        return cls(data=ReplyData(id=id))


class Face(BaseModel):
    """发送系统表情"""

    type: Literal["face"] = "face"
    data: FaceData

    @classmethod
    def new(cls, id: str):
        return cls(data=FaceData(id=id))


class Dice(BaseModel):
    """发送骰子"""

    type: Literal["dice"] = "dice"
    data: None | bool

    @classmethod
    def new(cls, data=None):
        return cls(data=data)


class Rps(BaseModel):
    """发送猜拳"""

    type: Literal["rps"] = "rps"
    data: None | bool

    @classmethod
    def new(cls, data=None):
        return cls(data=data)


class File(BaseModel):
    """发送文件"""

    type: Literal["file"] = "file"
    data: FileData

    @classmethod
    def new(cls, file: str):
        return cls(data=FileData(file=file))


class Video(BaseModel):
    """发送视频"""

    type: Literal["video"] = "video"
    data: VideoData

    @classmethod
    def new(cls, file: str):
        return cls(data=VideoData(file=file))


class Record(BaseModel):
    """发送语音"""

    type: Literal["record"] = "record"
    data: RecordData

    @classmethod
    def new(cls, file: str):
        return cls(data=RecordData(file=file))

type MessageSegment = (
    Text | At | Image | Reply | Face | Dice | Rps | File | Video | Record
)