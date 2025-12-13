from dishka import Provider, provide, Scope
from config import Settings,ModelParameterManager

class MyProvider(Provider):
    @provide(scope=Scope.APP)
    def get_config(self)->Settings:
        return Settings()
    
    @provide(scope=Scope.APP)
    def get_llm_setting(self,setting:Settings)->ModelParameterManager:
        llm_settings=setting.llm_settings
        return ModelParameterManager(llm_settings=llm_settings)
        
